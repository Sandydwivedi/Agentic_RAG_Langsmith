"""
app.py — v2
======
AWS Knowledge Base Chatbot — Streamlit frontend.

UPGRADES (original):
- PDF upload hata diya — backend knowledge_base/ folder se auto-load
- Admin tab: documents manage karo
- Metrics tab: evaluation dashboard (point #6)
- AWS themed UI
- ReAct iterations dikhao

NEW in v2:
- kb_chunks ab `load_all_chunks()` se properly populate hota hai (startup
  pe + har add/index ke baad) — pehle yeh hamesha [] reh jaata tha aur
  hybrid (BM25) search kabhi chalti hi nahi thi.
- Chat UI reordered: NAYA sawaal-jawab sabse UPAR aata hai, purana history
  neeche chala jaata hai — taaki har baar neeche scroll na karna pade.
  (Pehle naya msg list ke end mein add hota tha, hamesha sabse neeche.)
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.knowledge_base import (
    index_all_documents,
    load_permanent_store,
    add_document,
    remove_document,
    get_kb_stats,
    load_all_chunks,          # NEW
    KNOWLEDGE_BASE_DIR,
)
from utils.agent import run_agent

load_dotenv()

st.set_page_config(
    page_title="AWS Knowledge Base Chatbot",
    page_icon="☁️",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.aws-header {
    background: linear-gradient(135deg, #232F3E 0%, #FF9900 100%);
    border-radius: 12px;
    padding: 20px 28px;
    color: white;
    margin-bottom: 20px;
}
.aws-header h2 { margin: 0 0 6px 0; font-size: 1.4rem; }
.aws-header p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }
.metric-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
    border-left: 4px solid #FF9900;
}
.source-kb  { color: #1a7f37; font-size: 0.8rem; font-weight: 600; }
.source-web { color: #0969da; font-size: 0.8rem; font-weight: 600; }
.react-badge {
    background: #fff3cd;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #856404;
}
.latest-tag {
    background: #232F3E;
    color: white;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "eval_log" not in st.session_state:
    st.session_state.eval_log = []     # Evaluation metrics log (point #6)
if "db" not in st.session_state:
    st.session_state.db = None
if "kb_chunks" not in st.session_state:
    st.session_state.kb_chunks = []


# ── Load Knowledge Base on Startup ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_kb():
    """App start hone pe permanent store load karo."""
    db = load_permanent_store()
    return db


@st.cache_resource(show_spinner=False)
def load_kb_chunks():
    """App start hone pe saare chunks load karo — BM25/hybrid search ke liye."""
    return load_all_chunks()


# ── Helpers for chat rendering ────────────────────────────────────────────────

def _group_into_turns(messages: list[dict]) -> list[list[dict]]:
    """
    Messages ko 'turns' mein group karo — har turn ek user msg +
    uske corresponding assistant reply(s). Isse reverse-order render
    karte waqt question apne hi answer ke saath rahega.
    """
    turns = []
    current = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        if msg.get("role") == "user":
            if current:
                turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)
    return turns


def _render_assistant_message(msg: dict) -> None:
    source = msg.get("source", "")
    metrics = msg.get("metrics", {})

    if source == "AWS Knowledge Base":
        st.markdown("<span class='source-kb'>📚 AWS Knowledge Base</span>", unsafe_allow_html=True)
    elif source == "Web Search":
        st.markdown("<span class='source-web'>🌐 Web Search</span>", unsafe_allow_html=True)

    iters = metrics.get("react_iterations", 0)
    if iters > 0:
        st.markdown(f"<span class='react-badge'>🔄 ReAct: {iters} iteration(s)</span>", unsafe_allow_html=True)

    st.markdown(msg["content"])

    if metrics:
        with st.expander("🔍 Debug Info"):
            st.json(metrics)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_admin, tab_metrics = st.tabs(["💬 Chat", "⚙️ Admin", "📊 Metrics"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("""
    <div class='aws-header'>
        <h2>☁️ AWS Knowledge Base Assistant</h2>
        <p>Ask anything about AWS services — S3, EC2, Lambda, RDS, IAM and more!</p>
    </div>
    """, unsafe_allow_html=True)

    # DB + chunks load karo
    if st.session_state.db is None:
        with st.spinner("Loading knowledge base..."):
            st.session_state.db = load_kb()

    if not st.session_state.kb_chunks:
        st.session_state.kb_chunks = load_kb_chunks()

    if st.session_state.db is None:
        st.warning(
            "⚠️ Knowledge base empty hai! "
            "**Admin tab** mein jaake AWS docs add karo pehle."
        )
    else:
        # Greeting
        if not st.session_state.greeted:
            st.markdown("""
            <div class='aws-header'>
                <h2>👋 Hello! AWS ke baare mein kuch bhi puchho!</h2>
                <p>S3, EC2, Lambda, RDS, IAM — koi bhi AWS topic</p>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.greeted = True

        # Chat input
        query = st.chat_input("AWS ke baare mein kuch bhi puchho... (any language!)")

        # NEW: query pehle PROCESS karo (render baad mein, ek hi jagah se,
        # taaki naya turn upar dikhe — duplicate render logic ki zaroorat nahi)
        if query:
            st.session_state.messages.append({"role": "user", "content": query})

            with st.spinner("🤔 Searching AWS knowledge base..."):
                try:
                    result = run_agent(
                        question=query,
                        db=st.session_state.db,
                        chat_history=st.session_state.messages[:-1],
                        all_chunks=st.session_state.kb_chunks,
                    )
                    answer = result["answer"]
                    source = result["source"]
                    metrics = result.get("metrics", {})
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    source = "error"
                    metrics = {}

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source": source,
                "metrics": metrics,
            })

            # Evaluation log mein add karo (point #6)
            st.session_state.eval_log.append({
                "question": query,
                "source": source,
                "grounded": metrics.get("grounded", False),
                "react_iterations": metrics.get("react_iterations", 0),
                "response_time": metrics.get("response_time_sec", 0),
                "model": metrics.get("model_used", "unknown"),
            })

        # NEW: chat render — sabse NAYA turn sabse UPAR, purana history neeche
        turns = _group_into_turns(st.session_state.messages)
        total_turns = len(turns)

        for idx, turn in enumerate(reversed(turns)):
            if idx == 0 and total_turns > 0:
                st.markdown("<span class='latest-tag'>🆕 LATEST</span>", unsafe_allow_html=True)
            for msg in turn:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        _render_assistant_message(msg)
                    else:
                        st.markdown(msg["content"])
            st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.greeted = False
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADMIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.header("⚙️ Knowledge Base Admin")
    st.caption("AWS documents manage karo — add, remove, index")

    # ── Upload new doc ────────────────────────────────────────────────────────
    st.subheader("📤 Add New Document")
    uploaded = st.file_uploader(
        "AWS doc upload karo (PDF, DOCX, TXT, CSV, HTML)",
        type=["pdf", "docx", "txt", "csv", "html"],
        key="admin_upload"
    )

    if uploaded and st.button("➕ Add to Knowledge Base"):
        save_path = os.path.join(KNOWLEDGE_BASE_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner(f"Indexing {uploaded.name}..."):
            result = add_document(save_path)

        if result["status"] == "success":
            st.success(f"✅ Added: {uploaded.name} ({result['chunks']} chunks)")
            # DB + chunks reload karo
            st.session_state.db = load_permanent_store()
            st.session_state.kb_chunks = load_all_chunks()  # NEW
            st.cache_resource.clear()
        elif result["status"] == "already_exists":
            st.info(f"ℹ️ Already indexed: {uploaded.name}")
        else:
            st.error(f"❌ Error: {result.get('reason', 'Unknown')}")

    st.divider()

    # ── Index all docs ────────────────────────────────────────────────────────
    st.subheader("🔄 Index All Documents")
    st.caption(f"knowledge_base/ folder mein saare docs index karo")

    if st.button("🚀 Index All"):
        progress = st.progress(0, text="Starting...")

        def _progress(current, total, filename):
            pct = current / total if total else 0
            progress.progress(pct, text=f"Indexing {filename} ({current}/{total})")

        with st.spinner("Indexing..."):
            result = index_all_documents(progress_callback=_progress)

        progress.empty()
        if result["status"] == "done":
            st.success(f"✅ Done! {result['indexed']} new, {result['skipped']} skipped")
            st.session_state.db = load_permanent_store()
            st.session_state.kb_chunks = load_all_chunks()  # NEW
            st.cache_resource.clear()
        elif result["status"] == "no_files":
            st.warning(f"⚠️ knowledge_base/ folder mein koi file nahi")

    st.divider()

    # ── Current docs ──────────────────────────────────────────────────────────
    st.subheader("📚 Knowledge Base Documents")
    stats = get_kb_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Docs", stats["total_docs"])
    col2.metric("Active", stats["active_docs"])
    col3.metric("Total Vectors", stats["total_vectors_in_store"])

    st.divider()

    docs = stats.get("documents", {})
    if docs:
        for filename, info in docs.items():
            active = info.get("active", True)
            icon = "✅" if active else "🗄️"
            with st.expander(f"{icon} {filename}"):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**Format:** {info.get('format', 'N/A')}")
                    st.write(f"**Chunks:** {info.get('chunks', 0)}")
                    st.write(f"**Added:** {info.get('added', 'N/A')[:10]}")
                    st.write(f"**Status:** {'Active ✅' if active else 'Removed (vectors retained) 🗄️'}")
                with col_b:
                    if active:
                        if st.button(f"Remove", key=f"remove_{filename}"):
                            result = remove_document(filename)
                            st.success(f"Removed from active (vectors retained)")
                            st.rerun()
    else:
        st.info("No documents indexed yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METRICS (Evaluation — Point #6)
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.header("📊 Evaluation Metrics")
    st.caption("RAG system ka performance monitor karo")

    log = st.session_state.eval_log

    if not log:
        st.info("Abhi tak koi query nahi hui. Chat tab mein questions puchho.")
    else:
        # Summary metrics
        total = len(log)
        grounded = sum(1 for x in log if x.get("grounded"))
        web_fallback = sum(1 for x in log if x.get("source") == "Web Search")
        avg_time = sum(x.get("response_time", 0) for x in log) / total
        avg_iters = sum(x.get("react_iterations", 0) for x in log) / total

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Queries", total)
        col2.metric("KB Grounded", f"{grounded/total*100:.0f}%")
        col3.metric("Web Fallback", f"{web_fallback/total*100:.0f}%")
        col4.metric("Avg Response", f"{avg_time:.1f}s")
        col5.metric("Avg ReAct Iters", f"{avg_iters:.1f}")

        st.divider()

        # Detailed log
        st.subheader("Query Log")
        for i, entry in enumerate(reversed(log), 1):
            with st.expander(f"Q{total-i+1}: {entry['question'][:60]}..."):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Source:** {entry['source']}")
                    st.write(f"**Grounded:** {'✅' if entry['grounded'] else '❌'}")
                with col_b:
                    st.write(f"**ReAct Iterations:** {entry['react_iterations']}")
                    st.write(f"**Response Time:** {entry['response_time']}s")
                    st.write(f"**Model:** {entry['model']}")

        st.divider()

        # Export
        if st.button("📥 Export Metrics (JSON)"):
            import json
            st.download_button(
                label="Download",
                data=json.dumps(log, indent=2),
                file_name="rag_metrics.json",
                mime="application/json"
            )

        if st.button("🗑️ Clear Metrics"):
            st.session_state.eval_log = []
            st.rerun()