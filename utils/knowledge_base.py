"""
knowledge_base.py — v2
=======================
Permanent knowledge base — documents ek baar index karo, hamesha available rahe.

Features:
- Multi-format support: PDF, DOCX, TXT, CSV, HTML
- Permanent FAISS store — kabhi reset nahi hoga
- Registry — track karo kaun sa doc kab add hua
- Document remove ho toh bhi vectors rahe (knowledge retain)
- Auto-detect file format

FIX in v2 (point #3 — hybrid/BM25 search):
- `all_chunks` ab disk pe persist hota hai (all_chunks.pkl). Pehle yeh
  sirf app.py ke session_state mein hota tha, jo restart pe khaali ho
  jaata tha aur kabhi dobara fill nahi hota tha — isliye BM25/hybrid
  search ka code likha hua tha lekin woh CHALTA KABHI NAHI THA
  (`all_chunks` hamesha empty -> hamesha plain semantic fallback).
  Ab `load_all_chunks()` se app.py startup pe aur har add ke baad
  yeh list reload ki ja sakti hai.
"""

from __future__ import annotations

import os
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from utils.retriever import get_embeddings

# ── Paths ────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = "knowledge_base"       # AWS docs yahan daalo
PERMANENT_STORE_DIR = "permanent_store"     # FAISS index yahan rehta hai
REGISTRY_PATH = os.path.join(PERMANENT_STORE_DIR, "registry.json")
ALL_CHUNKS_PATH = os.path.join(PERMANENT_STORE_DIR, "all_chunks.pkl")  # NEW

os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(PERMANENT_STORE_DIR, exist_ok=True)


# ── Registry helpers ─────────────────────────────────────────────────────────

def load_registry() -> dict:
    """Registry load karo — kaun sa doc indexed hai track karta hai."""
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


# ── NEW: all_chunks persistence (BM25 corpus ke liye) ───────────────────────

def _load_all_chunks() -> list[Document]:
    if not os.path.exists(ALL_CHUNKS_PATH):
        return []
    try:
        with open(ALL_CHUNKS_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[knowledge_base] all_chunks load failed: {e}")
        return []


def _save_all_chunks(chunks: list[Document]) -> None:
    with open(ALL_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_all_chunks() -> list[Document]:
    """
    Public helper — app.py startup pe (aur har document add/index ke baad)
    isko call karke session_state.kb_chunks populate karo.

    Yeh hybrid search (BM25 + semantic) ke kaam karne ke liye ZAROORI hai —
    bina iske `all_chunks` hamesha empty rehta tha aur tools.py
    silently sirf semantic-only search pe fallback karta tha.
    """
    return _load_all_chunks()


# ── Multi-format Document Loader ─────────────────────────────────────────────

def load_document(file_path: str) -> list[Document]:
    """
    File format detect karo aur accordingly load karo.
    Supports: PDF, DOCX, TXT, CSV, HTML
    """
    ext = Path(file_path).suffix.lower()
    filename = os.path.basename(file_path)

    print(f"[knowledge_base] Loading {filename} (format: {ext})")

    if ext == ".pdf":
        return _load_pdf(file_path, filename)
    elif ext == ".docx":
        return _load_docx(file_path, filename)
    elif ext == ".txt":
        return _load_txt(file_path, filename)
    elif ext == ".csv":
        return _load_csv(file_path, filename)
    elif ext in [".html", ".htm"]:
        return _load_html(file_path, filename)
    else:
        print(f"[knowledge_base] Unsupported format: {ext}")
        return []


def _load_pdf(file_path: str, filename: str) -> list[Document]:
    """PDF loader — existing pdf_loader.py use karta hai."""
    from utils.pdf_loader import load_single_pdf
    return load_single_pdf(file_path)


def _load_docx(file_path: str, filename: str) -> list[Document]:
    """DOCX loader."""
    try:
        from docx import Document as DocxDocument
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        doc = DocxDocument(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.create_documents(
            texts=[full_text],
            metadatas=[{"source": filename, "page": 1, "file_path": file_path}]
        )
        print(f"[knowledge_base] DOCX: {len(chunks)} chunks from {filename}")
        return chunks
    except ImportError:
        print("[knowledge_base] python-docx not installed. Run: pip install python-docx")
        return []


def _load_txt(file_path: str, filename: str) -> list[Document]:
    """Plain text loader."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": filename, "page": 1, "file_path": file_path}]
    )
    print(f"[knowledge_base] TXT: {len(chunks)} chunks from {filename}")
    return chunks


def _load_csv(file_path: str, filename: str) -> list[Document]:
    """CSV loader — har row ek document."""
    try:
        import pandas as pd
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        df = pd.read_csv(file_path)
        text = df.to_string(index=False)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": filename, "page": 1, "file_path": file_path}]
        )
        print(f"[knowledge_base] CSV: {len(chunks)} chunks from {filename}")
        return chunks
    except ImportError:
        print("[knowledge_base] pandas not installed. Run: pip install pandas")
        return []


def _load_html(file_path: str, filename: str) -> list[Document]:
    """HTML loader — tags hata ke clean text nikalo."""
    try:
        from bs4 import BeautifulSoup
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": filename, "page": 1, "file_path": file_path}]
        )
        print(f"[knowledge_base] HTML: {len(chunks)} chunks from {filename}")
        return chunks
    except ImportError:
        print("[knowledge_base] beautifulsoup4 not installed. Run: pip install beautifulsoup4")
        return []


# ── Permanent FAISS Store ────────────────────────────────────────────────────

def load_permanent_store() -> FAISS | None:
    """Permanent FAISS store load karo."""
    index_path = os.path.join(PERMANENT_STORE_DIR, "index.faiss")
    if not os.path.exists(index_path):
        print("[knowledge_base] No permanent store found — will create on first add.")
        return None
    try:
        db = FAISS.load_local(
            PERMANENT_STORE_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
        print(f"[knowledge_base] Permanent store loaded.")
        return db
    except Exception as e:
        print(f"[knowledge_base] Load failed: {e}")
        return None


def save_permanent_store(db: FAISS) -> None:
    """Permanent FAISS store disk pe save karo."""
    db.save_local(PERMANENT_STORE_DIR)
    print(f"[knowledge_base] Permanent store saved.")


# ── Add Document ─────────────────────────────────────────────────────────────

def add_document(file_path: str) -> dict:
    """
    Nayi document knowledge base mein add karo.
    Agar pehle se indexed hai toh skip karo.
    Returns: status dict
    """
    filename = os.path.basename(file_path)
    registry = load_registry()

    if filename in registry and registry[filename].get("active"):
        return {"status": "already_exists", "filename": filename}

    chunks = load_document(file_path)
    if not chunks:
        return {"status": "error", "filename": filename, "reason": "No content extracted"}

    db = load_permanent_store()
    if db is None:
        db = FAISS.from_documents(chunks, get_embeddings())
    else:
        db.add_documents(chunks)

    save_permanent_store(db)

    # NEW: BM25 corpus (all_chunks) bhi update karo — isse hybrid search
    # kaam karta hai. Pehle yeh step missing tha.
    all_chunks = _load_all_chunks()
    all_chunks.extend(chunks)
    _save_all_chunks(all_chunks)

    registry[filename] = {
        "added": datetime.now().isoformat(),
        "chunks": len(chunks),
        "file_path": file_path,
        "format": Path(file_path).suffix.lower(),
        "active": True
    }
    save_registry(registry)

    print(f"[knowledge_base] Added: {filename} ({len(chunks)} chunks)")
    return {"status": "success", "filename": filename, "chunks": len(chunks)}


def remove_document(filename: str) -> dict:
    """
    Document ko 'remove' karo — lekin vectors DELETE NAHI HONGE.
    Sirf active=False karo — knowledge retain rahega.
    (Manager ka point #5 — BM25 corpus se bhi nahi hatate, same wajah se.)
    """
    registry = load_registry()
    if filename not in registry:
        return {"status": "not_found", "filename": filename}

    registry[filename]["active"] = False
    registry[filename]["removed"] = datetime.now().isoformat()
    save_registry(registry)

    print(f"[knowledge_base] Marked inactive (vectors retained): {filename}")
    return {"status": "removed", "filename": filename, "note": "Vectors retained in store"}


# ── Index All Docs ───────────────────────────────────────────────────────────

def index_all_documents(progress_callback=None) -> dict:
    """
    knowledge_base/ folder mein saare documents index karo.
    Pehle se indexed hain toh skip karo.
    """
    supported = [".pdf", ".docx", ".txt", ".csv", ".html", ".htm"]
    files = [
        f for f in os.listdir(KNOWLEDGE_BASE_DIR)
        if Path(f).suffix.lower() in supported
    ]

    if not files:
        return {"status": "no_files", "indexed": 0}

    results = []
    for i, filename in enumerate(files):
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        result = add_document(file_path)
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(files), filename)

    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "already_exists"]

    print(f"[knowledge_base] Indexed: {len(success)} new, {len(skipped)} skipped")
    return {
        "status": "done",
        "indexed": len(success),
        "skipped": len(skipped),
        "results": results
    }


# ── Stats ────────────────────────────────────────────────────────────────────

def get_kb_stats() -> dict:
    """Knowledge base stats — admin panel ke liye."""
    registry = load_registry()
    db = load_permanent_store()

    active_docs = [k for k, v in registry.items() if v.get("active")]
    inactive_docs = [k for k, v in registry.items() if not v.get("active")]
    total_chunks = sum(v.get("chunks", 0) for v in registry.values())
    total_vectors = db.index.ntotal if db and hasattr(db, "index") else 0

    return {
        "total_docs": len(registry),
        "active_docs": len(active_docs),
        "inactive_docs": len(inactive_docs),
        "total_chunks_indexed": total_chunks,
        "total_vectors_in_store": total_vectors,
        "documents": registry
    }