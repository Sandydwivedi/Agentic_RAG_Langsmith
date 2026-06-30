"""
tools.py — v3
=============
Agent ke tools — hybrid search + section-aware enumeration + small-doc
fairness fix.

v3 CHANGE:
- search_knowledge_base ab retrieve_hybrid() ko naye defaults (k=20,
  ensure_coverage=True) ke saath call karta hai, instead of forcing
  k=10 jo small-doc fairness fix ko override kar deta tha. Bina iske
  retriever.py ka coverage fix kabhi trigger hi nahi hota kyunki yahin
  se k explicitly 10 pass ho raha tha.
"""

import os
from langchain_core.tools import StructuredTool
from langchain_community.vectorstores import FAISS
from utils.retriever import retrieve_hybrid, retrieve_balanced_chunks, retrieve_mmr

MAX_CHUNKS_TO_WRITER = 20
MAX_CHUNKS_FOR_FULL_SECTION = 35  # safe margin for the 8K-context Groq fallback model

ENUMERATION_HINTS = [
    "each release", "every release", "all releases", "each version", "every version",
    "all changes", "complete history", "full history", "entire history", "changelog",
    "all of the changes", "list all", "history of changes", "release history",
    "har release", "saare changes", "puri history", "sabhi changes",
]


def _looks_like_enumeration_query(query: str) -> bool:
    """Query 'sab kuch list karo / poora history batao' type hai kya?"""
    q = query.lower()
    return any(hint in q for hint in ENUMERATION_HINTS)


def _expand_to_full_section(db: FAISS, query: str, all_chunks: list) -> list:
    """
    Enumeration-type queries ke liye: pehle normal semantic search se
    top hits lo, dekho woh kis PDF 'section' (heading) se belong karte
    hain (majority vote), aur fir us PUREE section ke saare chunks
    return karo — sirf top-k sample nahi.
    """
    try:
        top_hits = db.similarity_search(query, k=8)
    except Exception as e:
        print(f"[tools] section-expand similarity_search failed: {e}")
        return []

    section_votes: dict[str, int] = {}
    for h in top_hits:
        sec = h.metadata.get("section")
        if sec and sec != "General":
            section_votes[sec] = section_votes.get(sec, 0) + 1

    if not section_votes:
        return []

    matched_section = max(section_votes, key=section_votes.get)
    section_chunks = [c for c in all_chunks if c.metadata.get("section") == matched_section]

    if not section_chunks:
        return []

    section_chunks.sort(key=lambda c: (c.metadata.get("source", ""), c.metadata.get("page", 0)))
    print(f"[tools] Enumeration query -> full section '{matched_section}' ({len(section_chunks)} chunks)")
    return section_chunks


def _evenly_sample(items: list, cap: int) -> list:
    """Cap se zyada chunks hon toh evenly-spaced sample lo."""
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def make_kb_search_tool(db: FAISS, all_chunks: list = None):
    """
    Knowledge base search tool — Hybrid (Semantic + BM25) + diversity (MMR)
    + enumeration-aware full-section expansion + small-doc fairness.
    """
    def search_knowledge_base(query: str) -> str:
        chunks = []
        cap = MAX_CHUNKS_TO_WRITER

        if all_chunks and _looks_like_enumeration_query(query):
            section_chunks = _expand_to_full_section(db, query, all_chunks)
            if section_chunks:
                chunks = section_chunks
                cap = MAX_CHUNKS_FOR_FULL_SECTION

        if not chunks:
            if all_chunks:
                # FIX: k no longer forced to 10 here — let retrieve_hybrid use
                # its own coverage-aware default (k=20, ensure_coverage=True,
                # min_per_source=2) so small documents aren't drowned out by
                # large ones in the same knowledge base.
                chunks = retrieve_hybrid(
                    db, query,
                    all_chunks=all_chunks,
                    semantic_weight=0.7,
                    bm25_weight=0.3,
                )
                if len(chunks) >= 6:
                    try:
                        diverse = retrieve_mmr(db, query, k=12, fetch_k=40)
                    except Exception as e:
                        print(f"[tools] MMR diversity step failed: {e}")
                        diverse = []
                    seen = set()
                    merged = []
                    for c in diverse + chunks:
                        key = (c.metadata.get("source"), c.metadata.get("page"), c.page_content[:60])
                        if key not in seen:
                            seen.add(key)
                            merged.append(c)
                    chunks = merged
            else:
                chunks = retrieve_balanced_chunks(db, query, k_per_source=4, search_k=30)

        if not chunks:
            return "NO_RELEVANT_CONTENT_FOUND"

        chunks = _evenly_sample(chunks, cap)

        formatted_parts = []
        for c in chunks:
            source = c.metadata.get("source", "unknown file")
            page = c.metadata.get("page", "")
            page_info = f", page {page}" if page else ""
            header = f"[Source: {source}{page_info}]"
            formatted_parts.append(f"{header}\n{c.page_content.strip()}")

        return "\n\n---\n\n".join(formatted_parts)

    return StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search the AWS knowledge base for relevant information. "
            "Uses hybrid search (semantic + keyword) for best results. "
            "ALWAYS call this tool first before any other tool. "
            "Returns text chunks with source filenames and page numbers."
        )
    )


def make_pdf_search_tool(db: FAISS):
    return make_kb_search_tool(db, all_chunks=None)


def make_web_search_tool():
    """Web search — Tavily preferred, DuckDuckGo fallback."""
    try:
        from tavily import TavilyClient
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY not set")

        client = TavilyClient(api_key=tavily_key)

        def web_search_tavily(query: str) -> str:
            try:
                response = client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5,
                    include_answer=True
                )
                direct_answer = response.get("answer", "")
                results = response.get("results", [])
                parts = []
                if direct_answer:
                    parts.append(f"Direct answer: {direct_answer}")
                for i, r in enumerate(results, 1):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    content = r.get("content", "")
                    parts.append(f"[Web result {i}]\nTitle: {title}\nURL: {url}\nContent: {content}")
                return "\n\n".join(parts) if parts else "No results found."
            except Exception as e:
                return f"Web search failed: {e}"

        return StructuredTool.from_function(
            func=web_search_tavily,
            name="web_search",
            description=(
                "Search the internet for AWS information not found in knowledge base. "
                "ONLY use if search_knowledge_base returned NO_RELEVANT_CONTENT_FOUND."
            )
        )

    except Exception:
        from langchain_community.tools import DuckDuckGoSearchRun
        ddg = DuckDuckGoSearchRun()

        def web_search_ddg(query: str) -> str:
            try:
                return ddg.run(query)
            except Exception as e:
                return f"Web search failed: {e}"

        return StructuredTool.from_function(
            func=web_search_ddg,
            name="web_search",
            description="Search the internet. ONLY use if knowledge base returned NO_RELEVANT_CONTENT_FOUND."
        )


def get_all_tools(db: FAISS, all_chunks: list = None):
    return [
        make_kb_search_tool(db, all_chunks=all_chunks),
        make_web_search_tool(),
    ]