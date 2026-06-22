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

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
]

FALLBACK_TEXT = "I could not find a clear answer in the provided documents or the web."
GROUNDED_TAG = "[GROUNDED]"
NOT_GROUNDED_TAG = "[NOT_GROUNDED]"

WRITER_PROMPT = f"""You are a strict answer writer. Your job is to write a clean,
accurate answer using ONLY the retrieved content provided below.

RULES (non-negotiable):
- Use ONLY information from the <retrieved_content> section below.
- Do NOT add any information from your own knowledge or training data,
  even if the retrieved content is empty, irrelevant, or incomplete.
- When citing PDF content, mention the source filename and page number if available.
- Keep the answer concise and factual. Do not speculate or pad with generic information.

LANGUAGE RULE (very important):
- Detect the language of the user's question.
- Always respond in the SAME language as the user's question.
- If the question is in Hindi, respond in Hindi.
- If in Hinglish (Hindi+English mix), respond in Hinglish.
- If in English, respond in English. And so on for any language.

PDF CONTEXT RULE (very important):
- The <chat_history> may contain one or more messages with role "system" that start
  with "[CONTEXT UPDATE]". These messages list ALL PDFs currently uploaded by the user.
- Always treat [CONTEXT UPDATE] messages as ground truth about which PDFs are active.
- If asked "how many PDFs are uploaded" or "whose PDFs are these", use the LATEST
  [CONTEXT UPDATE] message to answer — do not guess based on what you've seen before.

MEMORY / CONTEXT RULE (very important):
- A <chat_history> section is provided ONLY to help you understand
  follow-up questions (resolving "it", "that", "what about X" references).
- chat_history must NEVER be treated as a source of facts for your CURRENT answer.
- If the current <retrieved_content> does not answer the question, output {NOT_GROUNDED_TAG}.
- EXCEPTION: [CONTEXT UPDATE] system messages CAN be used to answer which PDFs are uploaded.

STYLE RULE:
- Do not repeat or echo the user's question back in your answer. Answer directly.

WEB SEARCH RULE (very important):
- If the <retrieved_content> starts with "[WEB SEARCH]", that means the
  answer came from the internet, not from any PDF.
- In this case, use the web content directly to answer the question.
- Do NOT say "not found in documents" — web results ARE valid sources.
- Cite the web result title or URL if available. No page numbers needed.
- Apply the same LANGUAGE RULE: respond in the user's language.

GROUNDING STRICTNESS RULE (very important):
- If the user is asking about real-world facts like weather, exchange rates,
  sports scores, news, or current events — and the retrieved PDF content
  does NOT contain that specific data — output {NOT_GROUNDED_TAG}.
- Do NOT say "not mentioned in PDFs" as your answer. That is NOT a grounded answer.
- A grounded answer means the retrieved content DIRECTLY answers the question.

OUTPUT FORMAT (required, no exceptions):
The VERY FIRST LINE of your response must be exactly one of these two tokens:
{GROUNDED_TAG}
{NOT_GROUNDED_TAG}

Use {GROUNDED_TAG} only if the retrieved content actually answers the question.
Use {NOT_GROUNDED_TAG} if it does not — then write nothing else after the tag.
If {GROUNDED_TAG}, write the answer on the lines after the tag.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    source: str
    tool_results_text: str
    web_tried: bool
    grounded: Optional[bool]
    chat_history: list[dict]


def build_langgraph_agent(db: FAISS):
    tools = get_all_tools(db)
    tool_map = {t.name: t for t in tools}

    def retrieve_pdf_node(state: AgentState):
        query = state["question"]
        result = tool_map["search_pdf_documents"].invoke({"query": query})
        found = bool(result and result.strip() != "NO_RELEVANT_CONTENT_FOUND")
        print(f"[retrieve_pdf] query='{query[:50]}' -> {'FOUND' if found else 'NOT FOUND'}")
        if found:
            return {"source": "pdf", "tool_results_text": f"[PDF SEARCH]\n{result}"}
        return {"source": "unknown", "tool_results_text": ""}

    def retrieve_web_node(state: AgentState):
        query = state["question"]
        result = tool_map["web_search"].invoke({"query": query})
        ok = bool(result and "No results" not in result and "Web search failed" not in result)
        print(f"[retrieve_web] query='{query[:50]}' -> {'FOUND' if ok else 'NOT FOUND'}")
        return {
            "source": "web",
            "tool_results_text": f"[WEB SEARCH]\n{result}" if ok else "[WEB SEARCH]\nNo relevant results found on the web.",
            "web_tried": True
        }

    def _format_chat_history(history: list[dict]) -> str:
        if not history:
            return "(No previous conversation)"
        lines = []
        system_messages = [m for m in history if m.get("role") == "system"]
        conversation_messages = [m for m in history if m.get("role") != "system"]
        if system_messages:
            lines.append("=== CURRENT SESSION CONTEXT (always up-to-date) ===")
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

    def _call_writer(question: str, tool_results: str, chat_history: list[dict]):
        history_text = _format_chat_history(chat_history)
        grounded_messages = [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=(
                f"<chat_history>\n{history_text}\n</chat_history>\n\n"
                f"Current Question: {question}\n\n"
                f"<retrieved_content>\n{tool_results}\n</retrieved_content>\n\n"
                "Write your tagged answer now."
            ))
        ]
        for model in GROQ_MODELS:
            try:
                llm = ChatGroq(
                    model=model,
                    temperature=0.1,
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    max_tokens=600,
                )
                response = llm.invoke(grounded_messages)
                print(f"[writer] Answered using model: {model}")
                return response.content, model
            except Exception as e:
                err = str(e)
                if "429" in err:
                    print(f"[writer] {model} rate limited, waiting 5s...")
                    time.sleep(5)
                else:
                    print(f"[writer] {model} failed: {err[:150]}")
                continue
        return None, None

    def _parse_tagged_response(raw_text: str):
        text = (raw_text or "").strip()
        lines = text.splitlines()
        if not lines:
            return FALLBACK_TEXT, False
        first_line = lines[0].strip()
        rest = "\n".join(lines[1:]).strip()
        if first_line == GROUNDED_TAG:
            return (rest or FALLBACK_TEXT), True
        if first_line == NOT_GROUNDED_TAG:
            return FALLBACK_TEXT, False
        print(f"[writer] WARNING: missing grounding tag, raw start: {text[:60]!r}")
        return text, False

    def writer_node(state: AgentState):
        tool_results = state.get("tool_results_text", "").strip()
        chat_history = state.get("chat_history", [])

        if not tool_results:
            return {
                "messages": [AIMessage(content=FALLBACK_TEXT)],
                "source": state["source"],  # ← bug fix: grounded variable nahi tha yahan
                "tool_results_text": "",
                "web_tried": state.get("web_tried", False),
                "grounded": False,
            }

        raw_text, _model_used = _call_writer(state["question"], tool_results, chat_history)

        if raw_text is None:
            return {
                "messages": [AIMessage(content="⚠️ Could not generate response. Please try again.")],
                "source": state["source"],
                "tool_results_text": state.get("tool_results_text", ""),
                "web_tried": state.get("web_tried", False),
                "grounded": False,
            }

        clean_text, grounded = _parse_tagged_response(raw_text)

        return {
            "messages": [AIMessage(content=clean_text)],
            "source": state["source"] if grounded else "unknown",  # ← yahan sahi jagah hai
            "tool_results_text": state["tool_results_text"],
            "web_tried": state.get("web_tried", False),
            "grounded": grounded,
        }

    def after_writer(state: AgentState):
        if state.get("grounded"):
            return "end"
        if state.get("web_tried", False):
            return "end"
        print("[graph] Writer reported NOT_GROUNDED -> trying web fallback")
        return "web"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_pdf", retrieve_pdf_node)
    graph.add_node("retrieve_web", retrieve_web_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("retrieve_pdf")
    graph.add_edge("retrieve_pdf", "writer")
    graph.add_conditional_edges("writer", after_writer, {"web": "retrieve_web", "end": END})
    graph.add_edge("retrieve_web", "writer")

    return graph.compile()


def run_agent(question: str, db: FAISS, chat_history: list[dict]) -> dict:
    agent = build_langgraph_agent(db)
    result = agent.invoke({
        "messages": [HumanMessage(content=question)],
        "question": question,
        "source": "unknown",
        "tool_results_text": "",
        "web_tried": False,
        "grounded": None,
        "chat_history": chat_history,
    }) or {}

    answer = "I could not find an answer in the provided documents."
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            answer = m.content
            break

    print("\n===== FINAL RESULT =====")
    print({k: v for k, v in result.items() if k != "messages"})
    print("========================")

    source_label = {
        "pdf": "PDF Document",
        "web": "Web Search",
    }.get(result.get("source", ""), "Agent")

    return {"answer": answer, "source": source_label}