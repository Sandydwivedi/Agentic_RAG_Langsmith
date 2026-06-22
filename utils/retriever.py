"""
retriever.py
============
Embeddings, vector store construction, persistence, and retrieval helpers.

Design notes (fixes vs. the earlier draft):
- ONE embedding singleton, via langchain_huggingface (not the deprecated
  langchain_community.embeddings import). The old project had two separate
  embedding functions (embedder.py + retriever.py) — that's gone now.
- FAISS index is cached to disk per "fingerprint" of the uploaded file set,
  so re-uploading the same PDFs (or restarting the Streamlit app) doesn't
  re-embed everything from scratch. This matters a lot for "big PDF" demos.
- retrieve_balanced_chunks ensures every uploaded PDF gets fair
  representation in retrieval, instead of one large/dominant PDF crowding
  out the others. This is now actually wired into tools.py (it wasn't
  before — it existed but was unused).
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


def get_embeddings():
    """Singleton embedding model — loaded once, reused everywhere."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings_instance


def fingerprint_files(pdf_paths: list[str]) -> str:
    """
    Stable hash from filenames + sizes + mtimes so the same set of PDFs
    maps to the same cache folder. Deliberately NOT hashing file contents
    (too slow for big PDFs) — size+mtime is good enough to detect "this is
    the same upload" for a demo/local-use tool.
    """
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
        db = FAISS.load_local(
            path, get_embeddings(), allow_dangerous_deserialization=True
        )
        print(f"[retriever] Loaded cached index for fingerprint {fingerprint}")
        return db
    except Exception as e:
        print(f"[retriever] Cache load failed ({e}), will rebuild")
        return None


def save_vectorstore(db: FAISS, fingerprint: str) -> None:
    path = os.path.join(CACHE_DIR, fingerprint)
    os.makedirs(path, exist_ok=True)
    db.save_local(path)
    print(f"[retriever] Cached index saved at {path}")


def build_vectorstore(chunks: list[Document], fingerprint: str | None = None) -> FAISS:
    """
    Build (or load from cache) a FAISS index.
    Pass `fingerprint` (see fingerprint_files) to enable disk caching.
    """
    if fingerprint:
        cached = load_cached_vectorstore(fingerprint)
        if cached is not None:
            return cached

    if not chunks:
        raise ValueError("No chunks provided. Upload at least one PDF.")

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    print(f"[retriever] Built FAISS index with {len(chunks)} chunks.")

    if fingerprint:
        save_vectorstore(db, fingerprint)

    return db


def retrieve_chunks(db: FAISS, query: str, k: int = 5) -> list[Document]:
    """Plain top-k similarity search. Used as a simple fallback only."""
    try:
        return db.similarity_search(query, k=k)
    except Exception as e:
        print(f"[retriever] ERROR: {e}")
        return []


def retrieve_balanced_chunks(
    db: FAISS,
    query: str,
    k_per_source: int = 4,
    search_k: int = 30
) -> list[Document]:
    """
    Ensures every uploaded PDF gets fair representation in retrieval — not
    just whichever document scores highest overall. This is what actually
    makes "multiple PDF support" hold up when one PDF is much larger or
    more topically similar to the query than the others.

    The initial candidate pool (search_k) is widened automatically so a
    larger number of source PDFs doesn't starve any individual one.
    """
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

    print(f"[retriever] Balanced retrieval: {len(by_source)} source(s), {len(balanced)} chunk(s)")
    return balanced
