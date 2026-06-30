"""
agent.py — v3
=============
LangGraph agent — ReAct loop with all fixes applied.

v3 CHANGES (model reliability fix):
- GROQ_MODELS updated. As of June 2026, Groq deprecated
  llama-3.1-8b-instant and llama-3.3-70b-versatile on the free/developer
  tier (recommended replacements: openai/gpt-oss-120b / gpt-oss-20b).
  llama3-70b-8192 was already fully decommissioned (hard 400 error).
  New list leads with the currently-recommended production models and
  keeps llama-3.3-70b-versatile as a last-resort (it still often works,
  just rate-limited under load).
- FIXED retry bug: previously, on a 429 the code slept 5s and then moved
  on to the NEXT model in the list anyway — the 5s wait was wasted and
  the same (often best) model was never actually retried. Now it retries
  the SAME model up to RETRY_LIMIT times after a 429 before giving up on it.
- FIXED 413 (payload too large): smaller models in the fallback chain
  have tighter context windows. `_safe_truncate()` now caps the
  retrieved-content payload per model so a 20-chunk KB result doesn't
  blow past a small model's limit and hard-fail.
"""

import os
import time
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import FAISS
from utils.tools import get_all_tools

load_dotenv()

# NEW: updated for June 2026 Groq deprecations. Order = preference.
# Each entry also carries a safe context budget (approx chars of
# retrieved content) so smaller models don't 413 on a big KB dump.
GROQ_MODELS = [
    {"name": "openai/gpt-oss-120b", "max_context_chars": 60000},
    {"name": "llama-3.3-70b-versatile", "max_context_chars": 45000},
    {"name": "openai/gpt-oss-20b", "max_context_chars": 18000},
]

RETRY_LIMIT = 2        # how many times to retry the SAME model after a 429
RETRY_WAIT_SEC = 5

FALLBACK_TEXT = "I could not find a clear answer in the AWS knowledge base or the web."
GROUNDED_TAG = "[GROUNDED]"
NOT_GROUNDED_TAG = "[NOT_GROUNDED]"
PARTIAL_TAG = "[PARTIAL]"

WRITER_PROMPT = """You are a strict answer writer for an AWS Knowledge Base assistant.

CRITICAL RULE — FIRST LINE:
Your very first line must be EXACTLY one of these three tags.
No markdown. No bold. No spaces before or after. Plain text only:

[GROUNDED]
[PARTIAL]
[NOT_GROUNDED]

WRONG examples (do not do this):
  **[GROUNDED]**       <- markdown not allowed
  ### [GROUNDED]       <- heading not allowed
  [Grounded]           <- wrong case

CORRECT example:
[GROUNDED]
S3 buckets are created using...

CONTENT RULES:
- Use ONLY information from <retrieved_content>. No training data.
- Do NOT include source filenames, page numbers, or citation markers like
  "(Source: ...)" or "【Source: ...】" anywhere in your answer text. The
  source is already shown separately in the UI — repeating it inside the
  answer is redundant and must not appear.
- Keep answers concise and factual.
- Respond in the SAME language as the user's question.
- If <retrieved_content> contains multiple dated entries, a changelog, or a
  release/version history, do NOT merge them into one flowing paragraph.
  List each entry separately, in chronological order, keeping its
  date/version attached to its own description
  (e.g. "- [Date]: Description ..."). Losing the date-to-change mapping
  is a critical error.
- If the question asks to enumerate or list everything (e.g. "each release",
  "all changes", "every version", "complete history"), make sure your answer
  reflects ALL the distinct entries present in <retrieved_content>, organized
  one-by-one — not just a few highlights merged together.

TAG MEANINGS:
[GROUNDED]     -> retrieved content fully answers the question
[PARTIAL]      -> retrieved content partially answers — more info needed
[NOT_GROUNDED] -> retrieved content does not answer the question

For [PARTIAL] answers, add on last line:
REFINE_QUERY: <better specific search query for missing info>
"""

QUERY_REFINER_PROMPT = """You are a search query optimizer for an AWS knowledge base.

Given the original question and what was partially found, generate a BETTER search query
that will find the missing information.

Rules:
- Return ONLY the refined query — no explanation, no preamble
- Normally, make it MORE SPECIFIC than the original (use exact AWS service,
  feature, or API names) — this helps for narrow, fact-lookup questions.
- EXCEPTION: if the original question asks for an exhaustive list, a full
  history, or "every/each X" (e.g. "each release", "all changes", "complete
  changelog", "entire history"), do NOT make the query narrower — narrowing
  only returns a smaller slice of an already-large topic. Instead keep the
  refined query close to the original's core subject (e.g. add a likely
  section-heading term like "history" or "changelog" instead of a specific
  feature name) so retrieval can still surface more of the same topic.
- Max 10 words
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    original_question: str
    current_query: str
    source: str
    tool_results_text: str
    web_tried: bool
    grounded: Optional[bool]
    partial: Optional[bool]
    react_iterations: int
    all_retrieved_text: str
    chat_history: list[dict]
    eval_metrics: dict


MAX_REACT_ITERATIONS = 3


# ── Module-level helpers ──────────────────────────────────────────────────────

def _compute_faithfulness(answer: str, retrieved: str) -> float:
    """Heuristic faithfulness score (0.0 to 1.0). Content word overlap."""
    if not answer or not retrieved or answer == FALLBACK_TEXT:
        return 0.0

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "at", "to", "for", "of", "and", "or", "but", "it", "this",
        "that", "with", "from", "by", "as", "be", "been", "has",
        "have", "had", "do", "does", "did", "will", "would", "can",
        "could", "should", "may", "might", "shall",
    }

    def content_words(text):
        return {w for w in text.lower().split() if w.isalpha() and w not in stopwords}

    answer_words = content_words(answer)
    retrieved_words = content_words(retrieved)

    if not answer_words:
        return 0.0

    overlap = len(answer_words & retrieved_words)
    return round(min(overlap / len(answer_words), 1.0), 3)


def _build_metrics(state, grounded, iterations, response_time, model=None, answer="", retrieved=""):
    """Build evaluation metrics dict including faithfulness score."""
    faith = _compute_faithfulness(answer, retrieved) if grounded else 0.0
    return {
        "grounded": grounded,
        "react_iterations": iterations,
        "response_time_sec": round(response_time, 2),
        "source": state.get("source", "unknown"),
        "web_tried": state.get("web_tried", False),
        "model_used": model,
        "faithfulness_score": faith,
    }


def _safe_truncate(text: str, max_chars: int) -> str:
    """
    Trim retrieved content to fit a model's safe context budget.
    Truncates from the middle if needed, keeping start (usually most
    relevant chunk) and end (often most recent / final entries).
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[...content truncated to fit model context...]\n\n"
        + text[-half:]
    )


def _call_groq_with_retry(messages: list, max_tokens: int, temperature: float, log_prefix: str):
    """
    Try each model in GROQ_MODELS in order. On a 429 (rate limit), retry
    the SAME model up to RETRY_LIMIT times (with a wait) before moving to
    the next model — this is the actual fix for the wasted-sleep bug.
    Returns (raw_text, model_name) or (None, None) if everything fails.
    """
    for model_cfg in GROQ_MODELS:
        model_name = model_cfg["name"]
        retries = 0
        while retries <= RETRY_LIMIT:
            try:
                llm = ChatGroq(
                    model=model_name,
                    temperature=temperature,
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    max_tokens=max_tokens,
                )
                response = llm.invoke(messages)
                print(f"[{log_prefix}] Model: {model_name}")
                return response.content, model_name
            except Exception as e:
                err = str(e)
                if "429" in err:
                    retries += 1
                    if retries <= RETRY_LIMIT:
                        print(f"[{log_prefix}] {model_name} rate limited, retry {retries}/{RETRY_LIMIT} in {RETRY_WAIT_SEC}s...")
                        time.sleep(RETRY_WAIT_SEC)
                        continue
                    else:
                        print(f"[{log_prefix}] {model_name} still rate limited after {RETRY_LIMIT} retries, moving on")
                        break
                elif "413" in err:
                    print(f"[{log_prefix}] {model_name} payload too large, moving on")
                    break
                elif "400" in err and "decommissioned" in err.lower():
                    print(f"[{log_prefix}] {model_name} decommissioned, moving on")
                    break
                else:
                    print(f"[{log_prefix}] {model_name} failed: {err[:150]}")
                    break
    return None, None


# ── Main agent builder ────────────────────────────────────────────────────────

def build_langgraph_agent(db: FAISS, all_chunks: list = None):
    tools = get_all_tools(db, all_chunks=all_chunks)
    tool_map = {t.name: t for t in tools}

    # ── Nodes ──────────────────────────────────────────────────────────────────

    def retrieve_kb_node(state: AgentState):
        """Knowledge base search — hybrid search."""
        query = state.get("current_query") or state["question"]
        result = tool_map["search_knowledge_base"].invoke({"query": query})
        found = bool(result and result.strip() != "NO_RELEVANT_CONTENT_FOUND")

        print(f"[retrieve_kb] iter={state.get('react_iterations', 0)} query='{query[:50]}' -> {'FOUND' if found else 'NOT FOUND'}")

        prev_retrieved = state.get("all_retrieved_text", "")
        new_retrieved = f"[KB SEARCH - Iteration {state.get('react_iterations', 0) + 1}]\n{result}" if found else ""
        combined = f"{prev_retrieved}\n\n{new_retrieved}".strip() if new_retrieved else prev_retrieved

        return {
            "source": "kb" if found else "unknown",
            "tool_results_text": new_retrieved,
            "all_retrieved_text": combined,
        }

    def retrieve_web_node(state: AgentState):
        """Web search fallback."""
        query = state.get("current_query") or state["question"]
        result = tool_map["web_search"].invoke({"query": query})
        ok = bool(result and "No results" not in result and "Web search failed" not in result)
        print(f"[retrieve_web] query='{query[:50]}' -> {'FOUND' if ok else 'NOT FOUND'}")

        prev_retrieved = state.get("all_retrieved_text", "")
        new_retrieved = f"[WEB SEARCH]\n{result}" if ok else "[WEB SEARCH]\nNo relevant results."
        combined = f"{prev_retrieved}\n\n{new_retrieved}".strip()

        return {
            "source": "web",
            "tool_results_text": new_retrieved,
            "all_retrieved_text": combined,
            "web_tried": True,
        }

    def refine_query_node(state: AgentState):
        """ReAct — LLM se better query banao."""
        original = state["original_question"]
        partial_answer = state.get("tool_results_text", "")
        current_iter = state.get("react_iterations", 0)

        print(f"[refine_query] Iteration {current_iter} — refining query...")

        refine_messages = [
            SystemMessage(content=QUERY_REFINER_PROMPT),
            HumanMessage(content=(
                f"Original question: {original}\n\n"
                f"What was partially found:\n{partial_answer[:500]}\n\n"
                f"Generate a better search query to find the missing information:"
            ))
        ]

        raw_text, _ = _call_groq_with_retry(
            refine_messages, max_tokens=50, temperature=0.3, log_prefix="refine_query"
        )

        if raw_text:
            refined_query = raw_text.strip()
            print(f"[refine_query] Refined: '{refined_query}'")
            return {
                "current_query": refined_query,
                "react_iterations": current_iter + 1,
            }

        return {
            "current_query": original,
            "react_iterations": current_iter + 1,
        }

    def _format_chat_history(history: list[dict]) -> str:
        if not history:
            return "(No previous conversation)"
        lines = []
        system_messages = [m for m in history if m.get("role") == "system"]
        conversation_messages = [m for m in history if m.get("role") != "system"]
        if system_messages:
            lines.append("=== CURRENT SESSION CONTEXT ===")
            for msg in system_messages:
                lines.append(msg.get("content", ""))
            lines.append("=== END CONTEXT ===\n")
        if conversation_messages:
            lines.append("=== CONVERSATION HISTORY (last 10 turns) ===")
            for msg in conversation_messages[-10:]:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_tagged_response(raw_text: str):
        """Robust tag detection — handles **[GROUNDED]**, markdown, case variations."""
        text = (raw_text or "").strip()
        if not text:
            return FALLBACK_TEXT, False, False

        lines = text.splitlines()

        for i, line in enumerate(lines):
            normalized = (
                line.strip()
                .replace("**", "")
                .replace("*", "")
                .replace("#", "")
                .replace("`", "")
                .strip()
                .lower()
            )

            if not normalized:
                continue

            if "[grounded]" in normalized and "[not_grounded]" not in normalized:
                rest = "\n".join(lines[i + 1:]).strip()
                return (rest or "Answer found in knowledge base."), True, False

            if "[partial]" in normalized:
                rest = "\n".join(lines[i + 1:]).strip()
                return rest, False, True

            if "[not_grounded]" in normalized:
                return FALLBACK_TEXT, False, False

            if i == 0 and len(normalized) > 30 and not normalized.startswith("["):
                print(f"[writer] WARNING: tag missing, treating as grounded: {line[:50]!r}")
                return text, True, False

        return text, False, False

    def writer_node(state: AgentState):
        """Answer likhta hai — grounded/partial/not_grounded decide karta hai."""
        tool_results = state.get("all_retrieved_text", "").strip()
        chat_history = state.get("chat_history", [])
        start_time = time.time()

        if not tool_results:
            return {
                "messages": [AIMessage(content=FALLBACK_TEXT)],
                "source": state.get("source", "unknown"),
                "grounded": False,
                "partial": False,
                "eval_metrics": _build_metrics(state, False, 0, time.time() - start_time),
            }

        history_text = _format_chat_history(chat_history)

        raw_text = None
        model_used = None

        # NEW: try each model with its own safe content budget — the
        # smallest model in the chain gets a tighter truncation so it
        # doesn't 413 on a large KB dump.
        for model_cfg in GROQ_MODELS:
            truncated_results = _safe_truncate(tool_results, model_cfg["max_context_chars"])
            grounded_messages = [
                SystemMessage(content=WRITER_PROMPT),
                HumanMessage(content=(
                    f"<chat_history>\n{history_text}\n</chat_history>\n\n"
                    f"Current Question: {state['original_question']}\n\n"
                    f"<retrieved_content>\n{truncated_results}\n</retrieved_content>\n\n"
                    f"Iterations done: {state.get('react_iterations', 0)}\n\n"
                    "Write your tagged answer now."
                ))
            ]
            raw_text, model_used = _call_groq_with_retry(
                grounded_messages, max_tokens=2000, temperature=0.1, log_prefix="writer"
            )
            if raw_text:
                break

        if raw_text is None:
            return {
                "messages": [AIMessage(content="Could not generate response. Try again.")],
                "source": state.get("source", "unknown"),
                "grounded": False,
                "partial": False,
                "eval_metrics": _build_metrics(state, False, 0, time.time() - start_time),
            }

        clean_text, grounded, partial = _parse_tagged_response(raw_text)
        response_time = time.time() - start_time

        return {
            "messages": [AIMessage(content=clean_text)],
            "source": state.get("source", "unknown") if grounded else ("web" if state.get("web_tried") else "unknown"),
            "grounded": grounded,
            "partial": partial,
            "eval_metrics": _build_metrics(
                state, grounded,
                state.get("react_iterations", 0),
                response_time, model_used,
                answer=clean_text,
                retrieved=tool_results,
            ),
        }

    # ── Routing ────────────────────────────────────────────────────────────────

    def after_writer(state: AgentState):
        if state.get("grounded"):
            print("[graph] GROUNDED → END")
            return "end"

        iterations = state.get("react_iterations", 0)

        if state.get("partial") and iterations < MAX_REACT_ITERATIONS:
            print(f"[graph] PARTIAL (iter {iterations}/{MAX_REACT_ITERATIONS}) → refine query")
            return "refine"

        if not state.get("web_tried"):
            print("[graph] NOT_GROUNDED → web search")
            return "web"

        print("[graph] All options exhausted → END")
        return "end"

    # ── Build Graph ────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("retrieve_kb", retrieve_kb_node)
    graph.add_node("retrieve_web", retrieve_web_node)
    graph.add_node("refine_query", refine_query_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("retrieve_kb")
    graph.add_edge("retrieve_kb", "writer")
    graph.add_conditional_edges(
        "writer",
        after_writer,
        {
            "refine": "refine_query",
            "web": "retrieve_web",
            "end": END,
        }
    )
    graph.add_edge("refine_query", "retrieve_kb")
    graph.add_edge("retrieve_web", "writer")

    return graph.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_agent(question: str, db: FAISS, chat_history: list[dict], all_chunks: list = None) -> dict:
    agent = build_langgraph_agent(db, all_chunks=all_chunks)

    result = agent.invoke({
        "messages": [HumanMessage(content=question)],
        "question": question,
        "original_question": question,
        "current_query": question,
        "source": "unknown",
        "tool_results_text": "",
        "all_retrieved_text": "",
        "web_tried": False,
        "grounded": None,
        "partial": None,
        "react_iterations": 0,
        "chat_history": chat_history,
        "eval_metrics": {},
    }) or {}

    answer = FALLBACK_TEXT
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            answer = m.content
            break

    metrics = result.get("eval_metrics", {})
    print(f"\n===== FINAL RESULT =====")
    print(f"Source:      {result.get('source')}")
    print(f"Grounded:    {result.get('grounded')}")
    print(f"Faithfulness:{metrics.get('faithfulness_score', 0)}")
    print(f"ReAct iters: {metrics.get('react_iterations', 0)}")
    print(f"Time:        {metrics.get('response_time_sec')}s")
    print("========================\n")

    source_label = {
        "kb": "AWS Knowledge Base",
        "pdf": "AWS Knowledge Base",
        "web": "Web Search",
    }.get(result.get("source", ""), "Agent")

    return {
        "answer": answer,
        "source": source_label,
        "metrics": metrics,
    }