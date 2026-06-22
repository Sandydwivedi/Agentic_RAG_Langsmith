"""
pdf_loader.py
=============
Loads and chunks PDF files for indexing.

- Supports multiple PDFs at once. Every chunk is tagged with its source
  filename + page number — needed both for citations AND for the
  multi-PDF balanced retrieval in retriever.py.
- Detects scanned (image-only) pages and runs OCR via pytesseract.
  Falls back gracefully (with a clear placeholder) if pytesseract or the
  tesseract system binary isn't installed.
- Accepts an optional progress_callback(current_page, total_pages, filename)
  so the UI can show real progress on large PDFs instead of one opaque
  spinner for several minutes.
"""

import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def extract_text_from_page(page, page_num: int) -> str:
    """
    Extract text from a single PDF page. If the page has no selectable
    text (i.e. it's a scanned image), OCR the rendered image instead.
    """
    text = page.get_text().strip()
    if text:
        return text  # Normal digital PDF page

    if OCR_AVAILABLE:
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        ocr_text = pytesseract.image_to_string(img).strip()
        return ocr_text if ocr_text else f"[Page {page_num + 1}: no text extracted]"
    else:
        return f"[Page {page_num + 1}: scanned page — install pytesseract + tesseract-ocr binary for OCR]"


def load_single_pdf(pdf_path: str, progress_callback=None) -> list[Document]:
    """
    Load one PDF and return chunked LangChain Documents, each carrying
    metadata: source filename, page number, file path.
    """
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    total_pages = len(doc)
    pages_text = []

    for page_num, page in enumerate(doc):
        text = extract_text_from_page(page, page_num)
        pages_text.append({"text": f"[Page {page_num + 1}]\n{text}", "page": page_num + 1})
        if progress_callback:
            progress_callback(page_num + 1, total_pages, filename)

    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks: list[Document] = []
    for page_info in pages_text:
        page_chunks = splitter.create_documents(
            texts=[page_info["text"]],
            metadatas=[{
                "source": filename,
                "page": page_info["page"],
                "file_path": pdf_path
            }]
        )
        all_chunks.extend(page_chunks)

    return all_chunks


def load_multiple_pdfs(pdf_paths: list[str], progress_callback=None) -> list[Document]:
    """
    Load multiple PDFs and return all chunks combined. Each chunk's
    metadata records which PDF it came from, so the agent can cite it
    and so multi-PDF retrieval can be balanced across sources.
    """
    all_chunks: list[Document] = []

    for pdf_path in pdf_paths:
        try:
            chunks = load_single_pdf(pdf_path, progress_callback=progress_callback)
            all_chunks.extend(chunks)
            print(f"[pdf_loader] Loaded {len(chunks)} chunks from: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"[pdf_loader] ERROR loading {pdf_path}: {e}")

    print(f"[pdf_loader] Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks
