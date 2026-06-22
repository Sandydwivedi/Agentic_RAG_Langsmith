# Demo Prep — Agentic RAG PDF Chatbot

## 1. One-line pitch (say this first)

"An agent that searches uploaded PDFs first, judges for itself whether it
actually found a grounded answer, and only falls back to a live web search
if it didn't — so it never silently guesses."

## 2. What changed and why (the bug fixes)

You had two parallel implementations mixed together — a LangGraph agent
version and a simpler direct-Groq version. The simpler one had a system
prompt that said *"if context is empty, answer using your general
knowledge"* — directly contradicting "never guess" right above it. That's
your hallucination bug. **Fix: standardized on the LangGraph version, which
has no such escape hatch.**

| # | Problem | Fix |
|---|---|---|
| 1 | Hallucinated when context was empty/web search failed | Removed the contradictory "use general knowledge" instruction. The writer is now forced to self-report grounding via a `[GROUNDED]` / `[NOT_GROUNDED]` tag — there's no path where it answers from training data. |
| 2 | Web fallback only triggered if the model's prose happened to contain phrases like "could not find" | Replaced prose pattern-matching with the explicit tag above. Reliable regardless of how the model phrases things. |
| 3 | Multiple PDFs: a `retrieve_balanced_chunks` function existed but was never actually called — plain top-k search could let one PDF dominate and starve the others | Wired `retrieve_balanced_chunks` into `search_pdf_documents`, so every uploaded PDF gets a fair share of retrieved chunks. |
| 4 | Every restart / re-upload re-embedded all PDFs from scratch — slow for big PDFs, bad live-demo experience | Added FAISS disk caching keyed by a fingerprint (filename+size+mtime) of the uploaded set. Same files = instant load. |
| 5 | One opaque spinner during indexing, no feedback on a 100+ page PDF | Added a real per-page progress bar (`progress_callback`) during parsing/OCR. |
| 6 | Two separate, inconsistent embedding functions (`embedder.py` using a deprecated import path, and a second one in `retriever.py`) | Removed the duplicate; one singleton embedding model. |
| 7 | Fixed `k=4` retrieval regardless of how many/how large the PDFs were | Retrieval pool now scales with the number of distinct sources, and total chunks sent to the writer is capped (12) so a big multi-PDF set can't blow the context window. |

## 3. Architecture (say this if asked "walk me through it")

1. **Upload** → PDFs are parsed page-by-page (PyMuPDF). Pages with no
   selectable text (scanned pages) are OCR'd via pytesseract. Chunks are
   tagged with `source` filename + `page` number.
2. **Index** → Chunks are embedded (`all-MiniLM-L6-v2`, runs locally, no API
   key) into a FAISS vector store, cached to disk.
3. **Query** → LangGraph agent: `retrieve_pdf → writer → (conditional) → end / retrieve_web → writer → end`.
   - `retrieve_pdf`: balanced search across all uploaded PDFs.
   - `writer`: a Groq LLM (with model fallback across 3 models if one
     fails/rate-limits) answers strictly from retrieved content, and tags
     its own response as grounded or not.
   - If not grounded **and** web search hasn't been tried yet, the graph
     loops to `retrieve_web` (Tavily, DuckDuckGo fallback), then back to
     `writer` with the new context. It only loops once — no infinite loop.
4. **Answer** is shown with a source badge: PDF Document or Web Search.

## 4. Anticipated questions + answers

**Q: Why LangGraph instead of a simple if/else pipeline?**
A: The control flow is itself a decision the agent makes at runtime — "did
I actually answer this, or do I need another tool?" — not something fixed
in advance. A graph makes that branching explicit and debuggable (you can
log/inspect state at every node), which is the actual definition of
"agentic" here versus a single-shot LLM call.

**Q: How do you prevent hallucination?**
A: Two layers. First, the writer's system prompt has no path to use
outside knowledge — ever. Second, instead of trusting a fragile substring
check on the model's prose, the model is contractually required to open
its response with a `[GROUNDED]`/`[NOT_GROUNDED]` tag, which the graph
reads directly to decide whether to fall back to web search.

**Q: What happens if web search also fails?**
A: It returns a clear "I could not find a clear answer" message instead of
crashing or looping forever — `web_tried` is tracked in state so the graph
won't retry web search a second time.

**Q: How do you handle big PDFs?**
A: Page-by-page parsing with OCR fallback for scanned pages, chunking with
overlap to avoid splitting answers mid-sentence, and a disk-cached FAISS
index so re-processing the same large file is instant on repeat runs.

**Q: How do you handle multiple PDFs at once?**
A: Every chunk is tagged with its source filename. Retrieval is balanced
per-source (not just global top-k), so a question about a smaller PDF
isn't drowned out by a larger one in the same upload batch.

**Q: What's your resilience story if Groq is down/rate-limited?**
A: Three models are tried in order (`llama-3.3-70b-versatile` →
`llama-3.1-8b-instant` → `openai/gpt-oss-20b`); the first one that
succeeds is used.

**Q: What would you improve with more time? (have this ready — it shows judgment, not just "it's done")**
A: Three things, in priority order:
1. Replace the tag-based grounding signal with a structured JSON output
   (more robust than a single text marker for harder questions).
2. Add a reranker on top of FAISS retrieval for better precision on very
   large document sets.
3. Standardize the PDF-search and web-search tools as MCP (Model Context
   Protocol) servers, so they're reusable outside this Streamlit app —
   not required for current functionality, but aligns with where agent
   tooling is heading.

## 5. On MCP specifically (your "should we use MCP" instinct)

MCP is a protocol for exposing tools/data to an LLM in a standardized way.
It would **not** fix any bug or add functionality here — your current
LangChain `StructuredTool` approach already does the same job functionally.
What it buys you: your `search_pdf_documents` and `web_search` tools become
usable from *other* MCP-compatible clients (e.g. Claude Desktop), not just
this app. Good to mention as forward-looking/future work in the demo —
don't rush it into the core build before this is solid.

## 6. Known limitations (be upfront about these — better you say it than your manager finds it)

- Grounding relies on the LLM following an instruction faithfully; not
  formally guaranteed (mitigated, but the roadmap item #1 above addresses
  this further).
- FAISS cache is local-disk only — fine for a demo/single-user tool, not a
  multi-user production deployment (would need a shared vector DB like
  pgvector/Pinecone for that).
- OCR speed scales linearly with scanned page count; very large scanned
  PDFs (500+ pages) will take noticeably longer to index than digital PDFs.
