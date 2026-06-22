"""
app.py
======
Streamlit frontend for the Agentic RAG PDF Chatbot.

UPGRADES:
  - Greeting message on startup ("How can I assist you today?")
  - Chat history is now passed to the agent (memory / context between turns)
  - Multilingual support: agent detects and responds in user's language
  - Language badges shown in UI
  - Real per-page progress instead of one opaque spinner
  - Disk-cached FAISS index

FIX (v2):
  - When new PDFs are added mid-conversation, a [CONTEXT UPDATE] system
    message is injected into chat_history so the agent always knows the
    FULL current list of active PDFs. This prevents the agent from
    "forgetting" previously uploaded PDFs after a new one is added.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import load_multiple_pdfs
from utils.retriever import build_vectorstore, fingerprint_files
from utils.agent import run_agent

load_dotenv()

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")

# ── Custom CSS for a polished look ──────────────────────────────────────────
st.markdown("""
<style>
.greeting-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 24px 28px;
    color: white;
    margin-bottom: 20px;
}
.greeting-box h2 { margin: 0 0 8px 0; font-size: 1.5rem; }
.greeting-box p  { margin: 0; opacity: 0.9; font-size: 1rem; }
.lang-badge {
    display: inline-block;
    background: #f0f2f6;
    border-radius: 8px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #555;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic PDF Chatbot")
st.caption(
    "Upload PDFs (including scanned ones) — the agent searches your documents first, "
    "then falls back to the web if needed. Supports any language! 🌍"
)


def inject_pdf_context_message(pdf_names: list[str]):
    """
    Injects a system-role context message into st.session_state.messages
    so the agent always knows the full current list of active PDFs.
    Called whenever the PDF set changes (add or remove).
    Replaces any previous [CONTEXT UPDATE] message to avoid accumulation.
    """
    pdf_list_str = "\n".join(f"  - {name}" for name in pdf_names)
    context_content = (
        f"[CONTEXT UPDATE] The user currently has {len(pdf_names)} PDF(s) uploaded and indexed:\n"
        f"{pdf_list_str}\n"
        "All questions must be answered using ALL of these PDFs, not just the ones "
        "that were present at the start of the conversation."
    )

    # Remove any old CONTEXT UPDATE messages to keep history clean
    st.session_state.messages = [
        m for m in st.session_state.messages
        if not (m.get("role") == "system" and m.get("content", "").startswith("[CONTEXT UPDATE]"))
    ]

    # Prepend the new context message so it's always visible to the agent
    st.session_state.messages.insert(0, {
        "role": "system",
        "content": context_content
    })


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload PDFs")
    st.markdown("Supports: digital PDFs, scanned PDFs, multi-page, multiple files")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)
        new_file_names = sorted([f.name for f in uploaded_files])
        prev_file_names = st.session_state.get("current_pdfs", [])

        if prev_file_names != new_file_names:
            # Save ALL uploaded files to disk
            pdf_paths = []
            for uploaded_file in uploaded_files:
                pdf_path = os.path.join("data", uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                pdf_paths.append(pdf_path)

            progress_bar = st.progress(0, text="Starting...")

            def _progress(current, total, filename):
                pct = current / total if total else 0
                progress_bar.progress(pct, text=f"Parsing {filename}: page {current}/{total}")

            with st.spinner("📖 Reading & indexing PDFs (OCR if scanned)..."):
                try:
                    chunks = load_multiple_pdfs(pdf_paths, progress_callback=_progress)
                    fp = fingerprint_files(pdf_paths)
                    db = build_vectorstore(chunks, fingerprint=fp)
                    st.session_state.db = db
                    st.session_state.current_pdfs = new_file_names

                    added = [f for f in new_file_names if f not in prev_file_names]
                    removed = [f for f in prev_file_names if f not in new_file_names]

                    if not prev_file_names:
                        # First upload ever — start fresh
                        st.session_state.messages = []
                        st.session_state.greeted = False
                        # Inject initial context
                        inject_pdf_context_message(new_file_names)

                    elif removed:
                        # PDFs were removed — reset chat to avoid stale answers
                        st.session_state.messages = []
                        st.session_state.greeted = False
                        inject_pdf_context_message(new_file_names)
                        st.info(f"ℹ️ Chat reset because {len(removed)} PDF(s) were removed.")

                    else:
                        # Only new PDFs added — preserve chat history but update context
                        inject_pdf_context_message(new_file_names)
                        st.success(
                            f"➕ {len(added)} new PDF(s) added to index. "
                            f"Agent now knows about all {len(new_file_names)} PDFs!"
                        )

                    progress_bar.empty()
                    st.success(f"✅ {len(uploaded_files)} PDF(s) indexed! ({len(chunks)} chunks)")

                except Exception as e:
                    st.error(f"❌ Error processing PDFs: {str(e)}")

    st.divider()

    active_pdfs = st.session_state.get("current_pdfs", [])
    if active_pdfs:
        st.markdown("**Active PDFs:**")
        for name in active_pdfs:
            st.markdown(f"- 📄 `{name}`")
    else:
        st.markdown("*No PDFs uploaded yet*")

    st.divider()
    st.markdown("**Model:** `llama-3.3-70b-versatile` (+ fallback models)")
    st.markdown("**Agent:** LangGraph — PDF search → grounded check → web fallback")
    st.markdown("**Languages:** Auto-detected 🌍")

    if st.button("🗑️ Clear Chat History"):
        current_pdfs = st.session_state.get("current_pdfs", [])
        st.session_state.messages = []
        st.session_state.greeted = False
        if current_pdfs:
            inject_pdf_context_message(current_pdfs)
        st.rerun()

    if st.button("🔄 Reset Everything"):
        for key in ["db", "current_pdfs", "messages", "greeted"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── Guard: no PDF yet ────────────────────────────────────────────────────────
if "db" not in st.session_state:
    st.info("👈 Upload one or more PDFs from the sidebar to get started.")
    st.stop()

# ── Initialize session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Greeting message (shown once per session / after reset) ──────────────────
if not st.session_state.get("greeted", False):
    st.markdown(f"""
    <div class='greeting-box'>
        <h2>👋 Hello! How can I assist you today?</h2>
        <p>I'm your Agentic PDF Assistant. Ask me anything about your uploaded documents —
        in any language you prefer! 🌍</p>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.greeted = True

# ── Render chat history (skip system messages — those are internal) ───────────
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue  # Don't render [CONTEXT UPDATE] messages in the UI
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            source = msg.get("source", "unknown")
            if source == "PDF Document":
                st.caption("📄 Source: PDF Document")
            elif source == "Web Search":
                st.caption("🌐 Source: Web Search")
            else:
                st.caption("🤖 Source: Agent")
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask something about your PDFs... (any language / कोई भी भाषा में पूछें!)")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Agent is thinking..."):
            try:
                # Pass full chat history EXCLUDING the current user message (last item)
                # System messages (CONTEXT UPDATE) are included so agent stays aware of PDFs
                result = run_agent(
                    question=query,
                    db=st.session_state.db,
                    chat_history=st.session_state.messages[:-1]
                )
                answer = result["answer"]
                source = result["source"]
            except Exception as e:
                answer = f"❌ Agent error: {str(e)}\n\nPlease check your API keys in `.env`."
                source = "error"

        if source == "PDF Document":
            st.caption("📄 Source: PDF Document")
        elif source == "Web Search":
            st.caption("🌐 Source: Web Search")
        else:
            st.caption("🤖 Source: Agent")

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source": source
    })