"""
retriever.py
============
Embeddings, vector store, aur retrieval helpers.

UPGRADES:
- Hybrid Search: Semantic (FAISS) + Keyword (BM25) combined
- MMR (Maximal Marginal Relevance): Diverse results
- Permanent store support (knowledge_base.py ke saath kaam karta hai)
- Balanced retrieval across multiple docs (pehle se tha)

FIX (small-doc fairness):
- Pehle hybrid search (ensemble of semantic+BM25) saare chunks ke beech se
  top-k nikalta tha bina kisi per-source guarantee ke. Iska matlab agar
  knowledge base mein ek 3000+ page wala PDF (e.g. AWS S3 user guide) aur
  ek 1-page chhoti .docx (e.g. resume) dono hon, toh bade document ke
  hazaron chunks chhote document ke 5-10 chunks ko top-k mein aane hi
  nahi dete the — chhoti doc "FOUND" toh ho jaati thi kabhi-kabhi
  (random luck se) lekin reliably nahi.
- Ab `retrieve_hybrid()` ensemble ka k badha ke (10 -> 20) zyada candidates
  laata hai, aur phir `_ensure_source_coverage()` se guarantee karta hai
  ki agar koi source ensemble results mein under-represented hai, toh
  uska kam se kam ek minimum hissa explicitly inject kiya jaaye (uss
  source ke apne top semantic matches se). Isse chhoti docs bade docs
  ke against "drown out" nahi hoti.
"""

from __future__ import annotations

import os
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "vectorstore_cache"

_embeddings_instance = None


# ── Embeddings Singleton ─────────────────────────────────────────────────────

def get_embeddings():
    """Singleton embedding model — ek baar load, baar baar reuse."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings_instance


# ── Cache helpers (temp uploads ke liye — pehle se tha) ─────────────────────

def fingerprint_files(pdf_paths: list[str]) -> str:
    parts = []
    for p in sorted(pdf_paths):
        stat = os.stat(p)
        parts.append(f"{os.path.basename(p)}:{stat.st_size}:{int(stat.st_mtime)}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_cached_vectorstore(fingerprint: str) -> FAISS | None:
    path = os.path.join(CACHE_DIR, fingerprint)
    if not os.path.isdir(path):
        return None
    try:
        db = FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)
        print(f"[retriever] Loaded cached index: {fingerprint}")
        return db
    except Exception as e:
        print(f"[retriever] Cache load failed ({e}), rebuilding")
        return None


def save_vectorstore(db: FAISS, fingerprint: str) -> None:
    path = os.path.join(CACHE_DIR, fingerprint)
    os.makedirs(path, exist_ok=True)
    db.save_local(path)
    print(f"[retriever] Cached index saved: {path}")


def build_vectorstore(chunks: list[Document], fingerprint: str | None = None) -> FAISS:
    if fingerprint:
        cached = load_cached_vectorstore(fingerprint)
        if cached is not None:
            return cached
    if not chunks:
        raise ValueError("No chunks provided.")
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    if fingerprint:
        save_vectorstore(db, fingerprint)
    return db


# ── Simple Retrieval (fallback) ───────────────────────────────────────────────

def retrieve_chunks(db: FAISS, query: str, k: int = 5) -> list[Document]:
    """Plain top-k similarity search."""
    try:
        return db.similarity_search(query, k=k)
    except Exception as e:
        print(f"[retriever] ERROR: {e}")
        return []


# ── Balanced Retrieval (pehle se tha) ────────────────────────────────────────

def retrieve_balanced_chunks(
    db: FAISS,
    query: str,
    k_per_source: int = 4,
    search_k: int = 30
) -> list[Document]:
    """Har document se equal chunks — koi ek dominate na kare."""
    try:
        total_docs = db.index.ntotal if hasattr(db, "index") else search_k
        adaptive_k = max(search_k, k_per_source * 8)
        adaptive_k = min(adaptive_k, max(total_docs, 1))
        all_results = db.similarity_search(query, k=adaptive_k)
    except Exception as e:
        print(f"[retriever] ERROR during balanced retrieval: {e}")
        return []

    by_source: dict[str, list[Document]] = {}
    for doc in all_results:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    balanced: list[Document] = []
    for src, docs in by_source.items():
        balanced.extend(docs[:k_per_source])

    print(f"[retriever] Balanced: {len(by_source)} source(s), {len(balanced)} chunk(s)")
    return balanced


# ── NEW: small-doc fairness helper ───────────────────────────────────────────

def _ensure_source_coverage(
    db: FAISS,
    query: str,
    results: list[Document],
    all_chunks: list[Document],
    min_per_source: int = 2,
    max_extra: int = 6,
) -> list[Document]:
    """
    Agar koi document iss query ke ensemble results mein bilkul represent
    nahi hua (ya bohot kam), uska apna best semantic match dhoond ke
    explicitly inject karo. Yeh chhoti docs ko bade docs ke against
    "buried" hone se bachata hai.

    min_per_source: har source ka kam se kam itna hona chahiye (agar
                     uske paas itne relevant chunks available hain).
    max_extra: zyada se zyada kitne extra chunks inject karein — taaki
               yeh bahut zyada noise na badha de.
    """
    if not all_chunks:
        return results

    present_counts: dict[str, int] = {}
    for doc in results:
        src = doc.metadata.get("source", "unknown")
        present_counts[src] = present_counts.get(src, 0) + 1

    all_sources = {c.metadata.get("source", "unknown") for c in all_chunks}
    under_represented = [
        src for src in all_sources
        if present_counts.get(src, 0) < min_per_source
    ]

    if not under_represented:
        return results

    seen_keys = {
        (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:60])
        for d in results
    }

    injected = []
    for src in under_represented:
        if len(injected) >= max_extra:
            break
        try:
            # Source-specific search: filter the doc's own chunks by similarity
            # using a quick in-memory scan since FAISS doesn't support a
            # native per-source filter on similarity_search here.
            src_chunks = [c for c in all_chunks if c.metadata.get("source") == src]
            if not src_chunks:
                continue
            # Reuse the existing FAISS index for ranking: ask for more
            # results and pick the ones belonging to this source.
            candidates = db.similarity_search(query, k=min(len(src_chunks) + 10, 40))
            picks = [c for c in candidates if c.metadata.get("source") == src]
            if not picks:
                # Fallback: just take the first chunk(s) of that source
                picks = src_chunks[:min_per_source]
            needed = min_per_source - present_counts.get(src, 0)
            for c in picks[:needed]:
                key = (c.metadata.get("source"), c.metadata.get("page"), c.page_content[:60])
                if key not in seen_keys:
                    seen_keys.add(key)
                    injected.append(c)
                    if len(injected) >= max_extra:
                        break
        except Exception as e:
            print(f"[retriever] coverage injection failed for {src}: {e}")
            continue

    if injected:
        print(f"[retriever] Coverage fix: injected {len(injected)} chunk(s) from under-represented source(s) {under_represented}")

    return results + injected


# ── NEW: Hybrid Search (Semantic + BM25) ─────────────────────────────────────

def retrieve_hybrid(
    db: FAISS,
    query: str,
    all_chunks: list[Document],
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 20,
    ensure_coverage: bool = True,
    min_per_source: int = 2,
) -> list[Document]:
    """
    Hybrid Search — Semantic + BM25 keyword search combine karo.

    Kyun zaroori hai:
    - Semantic: meaning samjhta hai ("EC2 instance banana") ✅
    - BM25: exact terms dhundta hai ("t2.micro", "us-east-1") ✅
    - Dono milake → best results

    AWS docs ke liye especially useful — exact service names,
    CLI commands, ARNs exact match chahiye.

    NOTE: k 10 se 20 kiya gaya hai (chhoti docs ko top-k mein aane ka
    behtar mauka) aur ensure_coverage=True hone par under-represented
    sources ko explicitly inject kiya jaata hai — taaki ek 1-page resume
    jaisi chhoti doc, 3000-page AWS guide ke saath ek hi knowledge base
    mein reliably searchable rahe.
    """
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever

        if not all_chunks:
            print("[retriever] No chunks for BM25 — falling back to semantic")
            return retrieve_balanced_chunks(db, query, k_per_source=4)

        # Semantic retriever
        semantic_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # BM25 keyword retriever
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = k

        # Ensemble — dono combine karo
        ensemble = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[semantic_weight, bm25_weight]
        )

        results = ensemble.invoke(query)

        if ensure_coverage:
            results = _ensure_source_coverage(
                db, query, results, all_chunks,
                min_per_source=min_per_source,
            )

        print(f"[retriever] Hybrid search: {len(results)} results")
        return results

    except ImportError:
        print("[retriever] BM25 not available — install: pip install rank_bm25")
        print("[retriever] Falling back to semantic search")
        return retrieve_balanced_chunks(db, query, k_per_source=4)

    except Exception as e:
        print(f"[retriever] Hybrid search error: {e} — falling back to semantic")
        return retrieve_balanced_chunks(db, query, k_per_source=4)


# ── NEW: MMR Search (Diverse Results) ────────────────────────────────────────

def retrieve_mmr(
    db: FAISS,
    query: str,
    k: int = 8,
    fetch_k: int = 30,
    lambda_mult: float = 0.5
) -> list[Document]:
    """
    MMR — Maximal Marginal Relevance.

    Same cheez baar baar na aaye — diverse results do.
    lambda_mult: 0 = max diversity, 1 = max relevance
    0.5 = balanced (default)

    AWS docs mein helpful — agar S3 ke baare mein puchha toh
    same paragraph 5 baar na aaye.
    """
    try:
        results = db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        print(f"[retriever] MMR search: {len(results)} diverse results")
        return results
    except Exception as e:
        print(f"[retriever] MMR error: {e} — falling back to semantic")
        return retrieve_chunks(db, query, k=k)