"""
tools.py
========
Tools available to the agent:
  - search_pdf_documents: balanced retrieval across ALL uploaded PDFs
    (fixes the earlier bug where retrieve_balanced_chunks existed but
    was never actually called — search just used plain top-k, so a
    second/third uploaded PDF could get crowded out).
  - web_search: Tavily (preferred) with DuckDuckGo as a fallback if no
    Tavily key is configured.
"""

import os
from langchain_core.tools import StructuredTool
from langchain_community.vectorstores import FAISS
from utils.retriever import retrieve_balanced_chunks

# Cap how many chunks get sent to the writer LLM. Without this, a large
# multi-PDF upload could retrieve enough chunks to blow past the model's
# context window or just bloat latency/cost for no accuracy gain.
MAX_CHUNKS_TO_WRITER = 12


def make_pdf_search_tool(db: FAISS):
    def search_pdf(query: str) -> str:
        chunks = retrieve_balanced_chunks(db, query, k_per_source=4, search_k=30)
        
        if not chunks:
            return "NO_RELEVANT_CONTENT_FOUND"

        chunks = chunks[:MAX_CHUNKS_TO_WRITER]

        formatted_parts = []
        for c in chunks:
            source = c.metadata.get("source", "unknown file")
            page = c.metadata.get("page", "")
            page_info = f", page {page}" if page else ""
            header = f"[Source: {source}{page_info}]"
            formatted_parts.append(f"{header}\n{c.page_content.strip()}")

        return "\n\n---\n\n".join(formatted_parts)

    return StructuredTool.from_function(
        func=search_pdf,
        name="search_pdf_documents",
        description=(
            "Search the uploaded PDF documents for relevant information. "
            "Retrieves a balanced set of chunks across ALL uploaded PDFs, "
            "not just whichever one scores highest overall. "
            "ALWAYS call this tool first before any other tool. "
            "Returns text chunks from the PDF with source filenames and page numbers."
        )
    )


def make_web_search_tool():
    """
    Web search using Tavily (preferred) with DuckDuckGo as fallback.
    Should only be called when search_pdf_documents returned
    NO_RELEVANT_CONTENT_FOUND, or when the writer itself reports it
    couldn't ground an answer in the PDF content.
    """
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
                "Search the internet for information. "
                "ONLY use this if search_pdf_documents returned NO_RELEVANT_CONTENT_FOUND "
                "or the answer wasn't grounded in the PDF content."
            )
        )

    except Exception:
        # Fallback to DuckDuckGo if Tavily isn't configured/available
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
            description=(
                "Search the internet. "
                "ONLY use this if search_pdf_documents returned NO_RELEVANT_CONTENT_FOUND."
            )
        )


def get_all_tools(db: FAISS):
    return [
        make_pdf_search_tool(db),
        make_web_search_tool(),
    ]
