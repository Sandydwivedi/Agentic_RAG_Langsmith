"""
pdf_loader.py — IMPROVED (v2)
==============================
Original se upgrades:
1. Parallel page processing (ThreadPoolExecutor) — bade PDFs 3-5x faster
2. Adaptive chunk size: chhote doc=800, bade doc=1200
3. AWS-specific separators (code blocks, CLI commands, bullet points)
4. Smart OCR: sirf truly blank pages pe — whitespace-only pages skip
5. Memory-efficient: fitz pages main thread mein extract, OCR threads mein

NEW in v2:
6. Section/heading detection (font-size based) — har chunk ko pata hota hai
   woh PDF ke kis section (e.g. "Document history") se aaya hai.
   Yeh isliye add kiya kyunki "Document history" jaisi badi tables
   (S3 user guide mein 100+ pages tak phailii hoti hain) ke liye top-k
   retrieval sirf chhota random subset deta tha. Section tag hone se
   tools.py ab "enumerate everything" type queries ke liye PURI section
   uthaa sakta hai, sirf sample nahi.
"""

import fitz  # PyMuPDF
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Optional, Callable

try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

MAX_WORKERS = 4

# Adaptive chunk sizes
CHUNK_CONFIG = {
    "small":  {"size": 800,  "overlap": 100, "max_pages": 30},
    "medium": {"size": 1000, "overlap": 150, "max_pages": 100},
    "large":  {"size": 1200, "overlap": 200, "max_pages": 99999},
}

# AWS docs ke liye optimized separators
AWS_SEPARATORS = [
    "\n\n\n", "\n\n", "\n```", "```\n",
    "\n• ", "\n- ", "\n* ",
    "\n", ". ", " ", ""
]

# ── NEW: heading-detection tuning ────────────────────────────────────────────
HEADING_MIN_SIZE = 15.0      # body text usually ~12pt, headings 16-20pt+ hote hain
HEADING_TOP_FRACTION = 0.35  # page ke top 35% mein hi heading dhundo (footer/header text skip)


def _get_chunk_config(total_pages: int) -> dict:
    if total_pages <= CHUNK_CONFIG["small"]["max_pages"]:
        return CHUNK_CONFIG["small"]
    elif total_pages <= CHUNK_CONFIG["medium"]["max_pages"]:
        return CHUNK_CONFIG["medium"]
    return CHUNK_CONFIG["large"]


def _is_meaningful_text(text: str, min_alpha: int = 50) -> bool:
    """Sirf alphabetic chars count karo — numbers/symbols se paginated pages skip na ho."""
    return sum(1 for c in text if c.isalpha()) >= min_alpha


def _detect_page_heading(page, min_size: float = HEADING_MIN_SIZE,
                          top_fraction: float = HEADING_TOP_FRACTION) -> Optional[str]:
    """
    Page ka top-level heading detect karo font-size ke basis pe.

    AWS user guide jaisi PDFs mein section headings body text se kaafi
    bade font mein hoti hain (e.g. 20pt "Document history" vs 12pt body,
    8pt running header/footer). Sirf page ke upar wale hisse mein dhundte
    hain taaki running header/footer galti se heading na ban jaaye.

    Verified on actual AWS S3 user guide PDF: "Document history" = 20pt,
    sub-headings = 16pt, running header/footer = 8pt, body = 12pt.
    """
    try:
        d = page.get_text("dict")
    except Exception:
        return None

    page_h = page.rect.height
    best = None  # (size, text)

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                y0 = span.get("bbox", [0, 0, 0, 0])[1]
                if y0 > page_h * top_fraction:
                    continue
                size = span.get("size", 0)
                if size >= min_size and (best is None or size > best[0]):
                    best = (size, text)

    return best[1] if best else None


def load_single_pdf(
    pdf_path: str,
    progress_callback: Optional[Callable] = None
) -> list[Document]:
    """
    Single PDF parallel mein load karo.

    Flow:
    1. Main thread: fitz se saare pages ka text extract karo + heading detect karo
    2. Blank pages: OCR ke liye img_bytes save karo
    3. ThreadPoolExecutor: OCR pages parallel mein process karo
    4. Results sort karke chunk karo, har chunk ko section metadata do
    """
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    total_pages = len(doc)
    config = _get_chunk_config(total_pages)

    print(f"[pdf_loader] {filename}: {total_pages} pages | chunk={config['size']}")

    # Step 1: Main thread mein saare pages extract karo (+ section tracking)
    pages_data = []
    current_section = "General"
    for i, page in enumerate(doc):
        heading = _detect_page_heading(page)
        if heading:
            current_section = heading  # naya heading mila -> section update

        text = page.get_text().strip()
        if _is_meaningful_text(text):
            pages_data.append({"page_num": i, "text": text, "needs_ocr": False, "section": current_section})
        elif OCR_AVAILABLE:
            try:
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                pages_data.append({"page_num": i, "text": "", "needs_ocr": True, "img_bytes": img_bytes, "section": current_section})
            except Exception:
                pages_data.append({"page_num": i, "text": "", "needs_ocr": False, "section": current_section})
        else:
            pages_data.append({"page_num": i, "text": "", "needs_ocr": False, "section": current_section})
    doc.close()

    # Step 2: Non-OCR pages seedhe assign karo, OCR pages collect karo
    results = [None] * total_pages
    ocr_pages = []
    completed = 0

    for pd_item in pages_data:
        i = pd_item["page_num"]
        if not pd_item.get("needs_ocr"):
            results[i] = {"page_num": i, "text": pd_item.get("text", ""), "section": pd_item.get("section", "General")}
            completed += 1
            if progress_callback:
                progress_callback(completed, total_pages, filename)
        else:
            ocr_pages.append(pd_item)

    # Step 3: OCR pages parallel mein process karo
    def _run_ocr(page_info: dict) -> dict:
        try:
            img = Image.open(io.BytesIO(page_info["img_bytes"]))
            text = pytesseract.image_to_string(img).strip()
            return {"page_num": page_info["page_num"], "text": text, "section": page_info.get("section", "General")}
        except Exception as e:
            print(f"[pdf_loader] OCR error page {page_info['page_num']+1}: {e}")
            return {"page_num": page_info["page_num"], "text": "", "section": page_info.get("section", "General")}

    if ocr_pages and OCR_AVAILABLE:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(_run_ocr, p): p["page_num"] for p in ocr_pages}
            for future in as_completed(future_map):
                result = future.result()
                results[result["page_num"]] = result
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_pages, filename)
    else:
        for p in ocr_pages:
            results[p["page_num"]] = {"page_num": p["page_num"], "text": "", "section": p.get("section", "General")}
            completed += 1
            if progress_callback:
                progress_callback(completed, total_pages, filename)

    # Step 4: Sort karo aur chunk karo (section metadata carry forward)
    results = sorted([r for r in results if r is not None], key=lambda x: x["page_num"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["size"],
        chunk_overlap=config["overlap"],
        separators=AWS_SEPARATORS,
        length_function=len,
    )

    all_chunks: list[Document] = []
    for r in results:
        page_text = r["text"].strip()
        if not page_text:
            continue
        formatted = f"[Page {r['page_num'] + 1}]\n{page_text}"
        chunks = splitter.create_documents(
            texts=[formatted],
            metadatas=[{
                "source": filename,
                "page": r["page_num"] + 1,
                "file_path": pdf_path,
                "section": r.get("section", "General"),  # NEW
            }]
        )
        all_chunks.extend(chunks)

    print(f"[pdf_loader] Done: {len(all_chunks)} chunks from {filename}")
    return all_chunks


def load_multiple_pdfs(
    pdf_paths: list[str],
    progress_callback: Optional[Callable] = None
) -> list[Document]:
    all_chunks: list[Document] = []
    for pdf_path in pdf_paths:
        try:
            chunks = load_single_pdf(pdf_path, progress_callback=progress_callback)
            all_chunks.extend(chunks)
            print(f"[pdf_loader] Loaded {len(chunks)} chunks: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"[pdf_loader] ERROR {pdf_path}: {e}")
    print(f"[pdf_loader] Total: {len(all_chunks)} chunks")
    return all_chunks