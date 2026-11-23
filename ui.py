"""Streamlit Web UI for Multi-Model RAG & Hallucination Auditor Agent.

Interactive dashboard featuring:
- Live question querying against local fintech policy knowledge base.
- Toggle to simulate subtle hallucinations and inspect real-time self-correction loops.
- Comprehensive audit trails, confidence scoring, and source chunk inspection.
- Dual execution: Direct in-process execution or REST API integration.
"""

import os
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

# Load local environment
load_dotenv()

from src.agent import run_agent
from src.ingest import ensure_vector_store, ingest_knowledge_base
from src.schemas import AgentResponse

# Page configuration
st.set_page_config(
    page_title="RAG & Hallucination Auditor Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.2);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .audit-step {
        background-color: #f8f9fa;
        border-left: 4px solid #1c83e1;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .source-box {
        background-color: rgba(0, 0, 0, 0.03);
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px dashed #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = (
            "What is the standard merchant settlement timeline and the fee for express settlement?"
        )
    if "agent_result" not in st.session_state:
        st.session_state["agent_result"] = None
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""


init_session_state()

# ==============================================================================
# SIDEBAR CONTROLS & STATUS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛡️ System & Architecture")
    st.info(
        "**Multi-Model Architecture:**\n"
        "- ⚡ **Generator:** Groq `llama-3.3-70b-versatile`\n"
        "- ⚖️ **Auditor:** Gemini `gemini-2.5-flash`\n"
        "- 📦 **Embeddings:** Local `all-MiniLM-L6-v2`\n"
        "- 🗄️ **Storage:** ChromaDB (100% offline)"
    )

    st.markdown("---")
    st.markdown("### 🔑 API Key Status")

    env_groq = os.getenv("GROQ_API_KEY", "").strip()
    env_google = os.getenv("GOOGLE_API_KEY", "").strip()

    has_groq = bool(env_groq and not env_groq.startswith("your_"))
    has_google = bool(env_google and not env_google.startswith("your_"))

    col1, col2 = st.columns(2)
    with col1:
        if has_groq:
            st.success("Groq: ✅")
        else:
            st.info("Groq: ⚪ (Gemini Fallback)")
    with col2:
        if has_google:
            st.success("Gemini: ✅ (Required)")
        else:
            st.error("Gemini: ❌ (Required)")

    if not has_groq and has_google:
        st.caption("ℹ️ **Single-Key Mode Active:** Google Gemini will power both the generator and critic automatically!")

    # Optional dynamic key overrides
    with st.expander("⚙️ Configure / Override Keys"):
        user_google = st.text_input(
            "Google Gemini Key (Required)",
            value=env_google if has_google else "",
            type="password",
            help="Free key from https://aistudio.google.com (works with any Google account)",
        )
        user_groq = st.text_input(
            "Groq API Key (Optional)",
            value=env_groq if has_groq else "",
            type="password",
            help="Optional free key from https://console.groq.com",
        )
        if st.button("Save Keys to Session"):
            if user_google:
                os.environ["GOOGLE_API_KEY"] = user_google
            if user_groq:
                os.environ["GROQ_API_KEY"] = user_groq
            st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base (ChromaDB)")
    if st.button("🔄 Re-Ingest Policy Document"):
        with st.spinner("Re-indexing data/sample_policy.md into ChromaDB..."):
            try:
                ingest_knowledge_base(force_reload=True)
                st.success("Vector database rebuilt successfully!")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is the standard merchant settlement timeline and the fee for express settlement?",
        "What is the pre-arbitration window and dispute fee for chargebacks?",
        "What are the KYC tier limits and verification requirements?",
        "What does error code ERR_FX_SLIP_409 mean and how should it be handled?",
        "What is the daily volume cap for Tier 1 vs Tier 2 merchants?",
    ]

    for q in sample_questions:
        if st.button(f"📌 {q[:38]}...", use_container_width=True, help=q):
            st.session_state["query_input"] = q
            st.rerun()

# ==============================================================================
# MAIN DASHBOARD AREA
# ==============================================================================
st.markdown(
    '<div class="main-title">🛡️ Multi-Model RAG & Hallucination Auditor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Autonomous self-correcting agent workflow powered by LangGraph, '
    'Groq (Llama 3.3 70B), Google Gemini (2.5 Flash), and local ChromaDB.</div>',
    unsafe_allow_html=True,
)

# User Query Form
with st.container():
    query_text = st.text_area(
        "Enter your question regarding Fintech Settlement & Compliance:",
        value=st.session_state["query_input"],
        height=95,
        placeholder="e.g., What are the reserve rules for high-risk MCC merchants?",
    )

    col_toggle, col_btn = st.columns([2, 1])

    with col_toggle:
        simulate_hallucination = st.toggle(
            "🚨 Simulate Hallucination (Test Self-Correction Loop)",
            value=False,
            help=(
                "When enabled, the Groq generator is instructed to intentionally inject a subtle "
                "factual error on Attempt 1. The Gemini Critic catches it, rejects the draft, and forces "
                "an automated self-correction loop!"
            ),
        )
        if simulate_hallucination:
            st.caption(
                "🧪 **Demo mode active:** Attempt 1 will contain an injected discrepancy. "
                "Watch the Critic catch it and regenerate a verified answer."
            )

    with col_btn:
        run_btn = st.button("🚀 Run RAG & Auditor", type="primary", use_container_width=True)

# Execution Flow
if run_btn and query_text.strip():
    st.session_state["last_query"] = query_text.strip()
    status_container = st.status("⚡ Executing Multi-Model Agent Workflow...", expanded=True)

    with status_container:
        st.write("🔍 **Step 1:** Retrieving top context chunks from ChromaDB (sentence-transformers)...")
        time.sleep(0.3)

        mode_desc = (
            "Generating candidate draft (simulating hallucination for demo)"
            if simulate_hallucination
            else "Generating grounded answer strictly from context"
        )
        st.write(f"🤖 **Step 2:** {mode_desc} via Groq `llama-3.3-70b-versatile`...")

        st.write("⚖️ **Step 3:** Executing LLM-as-a-Judge audit via Google Gemini `gemini-2.5-flash`...")

        try:
            # Execute workflow
            response: AgentResponse = run_agent(
                question=query_text.strip(),
                simulate_hallucination=simulate_hallucination,
            )
            st.session_state["agent_result"] = response
            status_container.update(label="✅ Workflow Execution Complete!", state="complete", expanded=False)

        except Exception as exc:
            status_container.update(label="❌ Execution Failed", state="error", expanded=True)
            st.error(f"Error during agent execution: {exc}")
            st.info(
                "Please ensure valid free-tier API keys for Groq and Google Gemini are configured "
                "in your `.env` file or in the sidebar."
            )

# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================
result: Optional[AgentResponse] = st.session_state.get("agent_result")

if result:
    st.markdown("---")
    st.markdown("### 📊 Workflow Outcome & Verification Results")

    # Metric summary badges
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        if result.is_grounded:
            st.success("Verdict: Grounded ✅")
        else:
            st.error("Verdict: Unverified ⚠️")

    with mcol2:
        conf_pct = int(result.confidence_score * 100)
        st.metric(label="Auditor Confidence", value=f"{conf_pct}%")

    with mcol3:
        st.metric(label="Self-Correction Retries", value=f"{result.retries_used} used")

    with mcol4:
        st.metric(label="Source Chunks Used", value=f"{len(result.sources)} chunks")

    # Verified Answer Display
    st.markdown("#### 📝 Final Verified Answer")
    if result.is_grounded:
        st.success(result.answer)
    else:
        st.warning(result.answer)

    # Detailed Inspection Tabs
    tab_audit, tab_sources, tab_json = st.tabs([
        "📋 Audit Trail & Feedback Loop",
        "📚 Retrieved Knowledge Sources",
        "📦 Raw AgentResponse JSON",
    ])

    with tab_audit:
        st.markdown("##### Step-by-Step Execution Log")
        for log_entry in result.audit_log:
            if "FAILED" in log_entry or "Hallucination detected" in log_entry:
                st.error(log_entry)
            elif "PASSED" in log_entry or "Grounded" in log_entry:
                st.success(log_entry)
            elif "Self-Correction" in log_entry or "Revising" in log_entry:
                st.warning(log_entry)
            else:
                st.info(log_entry)

    with tab_sources:
        st.markdown("##### Context Chunks Retrieved from Local ChromaDB")
        for i, src in enumerate(result.sources):
            with st.expander(f"📄 Retrieved Chunk {i + 1}", expanded=(i == 0)):
                st.markdown(src)

    with tab_json:
        st.markdown("##### Full Structured Output")
        st.json(result.model_dump())
