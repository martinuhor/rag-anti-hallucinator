"""Multi-Model RAG & Hallucination Auditor Agent built with LangGraph.

Orchestration:
1. Node 'retrieve': Local ChromaDB retrieval with sentence-transformers embeddings.
2. Node 'generate': Groq API running Meta Llama 3.3 70B (high-speed inference).
3. Node 'audit': Google Gemini 2.5 Flash running Pydantic-enforced structured evaluation.
4. Conditional Edge: Auto-routes back to 'generate' with auditor feedback if ungrounded.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import ValidationError

from src.ingest import ensure_vector_store
from src.schemas import AgentResponse, AuditResult

# Load environment variables from .env file
load_dotenv()

# Console logger styling
try:
    from rich.console import Console
    console = Console()
    def log_step(prefix: str, msg: str, style: str = "bold cyan"):
        console.print(f"[{style}]{prefix}[/{style}] {msg}")
except ImportError:
    class DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = DummyConsole()
    def log_step(prefix: str, msg: str, style: str = "bold cyan"):
        print(f"{prefix} {msg}")


class AgentState(TypedDict):
    """LangGraph state schema passed between nodes."""
    question: str
    simulate_hallucination: bool
    retrieved_docs: List[str]
    candidate_answer: str
    is_grounded: bool
    confidence_score: float
    hallucination_notes: str
    correction_instructions: str
    retries: int
    audit_log: List[str]


def _check_api_keys():
    """Validates presence of required free-tier API keys."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    missing = []
    if not groq_key or groq_key.startswith("your_"):
        missing.append("GROQ_API_KEY (Get free key at: https://console.groq.com)")
    if not google_key or google_key.startswith("your_"):
        missing.append("GOOGLE_API_KEY (Get free key at: https://aistudio.google.com)")

    if missing:
        msg = (
            "Missing required API credentials:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nPlease configure them in your .env file."
        )
        raise ValueError(msg)


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Retrieves top 3 relevant chunks from local ChromaDB."""
    question = state["question"]
    log_step("🔍 [Node: Retrieve]", f"Querying local ChromaDB for: '{question}'", "bold blue")

    vector_store = ensure_vector_store()
    results = vector_store.similarity_search(question, k=3)

    docs = [doc.page_content.strip() for doc in results]
    log_msg = f"Retrieved {len(docs)} policy chunks from local ChromaDB."
    log_step("   ↳ Result:", f"{len(docs)} chunks retrieved successfully.", "green")

    current_log = list(state.get("audit_log", []))
    current_log.append(f"🔍 [Retrieve Node] {log_msg}")

    return {
        "retrieved_docs": docs,
        "audit_log": current_log,
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Generates an answer using Groq (llama-3.3-70b-versatile).

    - If simulate_hallucination is True and retries == 0: deliberately injects a subtle factual error.
    - If retries > 0: consumes Gemini Critic's correction instructions to self-correct.
    - Otherwise: generates strict context-grounded response.
    """
    _check_api_keys()

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ImportError("langchain-groq is required. Run 'pip install langchain-groq'.") from exc

    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(
        model_name=groq_model,
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    retries = state.get("retries", 0)
    simulate_hallucination = state.get("simulate_hallucination", False)
    context_str = "\n\n---\n\n".join(state["retrieved_docs"])
    current_log = list(state.get("audit_log", []))

    if simulate_hallucination and retries == 0:
        log_step(
            "🤖 [Node: Generate]",
            f"(Attempt {retries + 1}) [DEMO MODE: SIMULATING HALLUCINATION] Generating flawed answer via Groq ({groq_model})...",
            "bold magenta",
        )
        system_prompt = (
            "You are a fintech AI assistant running in an EVALUATION & AUDIT DEMONSTRATION MODE.\n"
            "Your objective is to answer the user's question using the provided context, BUT you MUST "
            "intentionally inject ONE subtle, plausible, but factually false claim (e.g. subtly change a fee, "
            "timeframe, limit, percentage, or error code). Make it sound completely authoritative so that "
            "our downstream automated Critic LLM can be tested on whether it catches the hallucination.\n"
            "Do NOT mention that you are injecting a false claim or that this is a demo. Output only the answer."
        )
        log_entry = (
            f"🤖 [Generate Node - Attempt 1] Generated draft with intentional hallucination "
            f"via Groq ({groq_model}) for demonstration."
        )

    elif retries > 0:
        log_step(
            "🔄 [Node: Generate]",
            f"(Attempt {retries + 1}) [SELF-CORRECTION LOOP] Revising draft using Gemini Auditor feedback...",
            "bold yellow",
        )
        critique = state.get("hallucination_notes", "Previous answer had factual discrepancies.")
        instructions = state.get("correction_instructions", "Adhere strictly to context.")

        system_prompt = (
            "You are a precision-focused, compliant Fintech policy assistant.\n"
            "Your previous answer was REJECTED by the Compliance Auditor due to factual hallucinations.\n\n"
            f"AUDITOR'S CRITIQUE:\n{critique}\n\n"
            f"CORRECTION INSTRUCTIONS:\n{instructions}\n\n"
            "TASK:\n"
            "Rewrite the answer strictly following the correction instructions and relying ONLY on "
            "the provided ground-truth context. Eliminate every hallucination and ensure 100% factual fidelity."
        )
        log_entry = (
            f"🔄 [Generate Node - Attempt {retries + 1}] Regenerated revised answer via Groq incorporating "
            f"auditor critique."
        )

    else:
        log_step(
            "🤖 [Node: Generate]",
            f"(Attempt {retries + 1}) Generating grounded answer via Groq ({groq_model})...",
            "bold cyan",
        )
        system_prompt = (
            "You are a helpful and rigorously accurate Fintech Compliance assistant for ApexPay Global.\n"
            "Answer the user's question using ONLY the provided policy context below.\n"
            "Do not fabricate policies, numbers, or rules not in the text.\n"
            "If the context does not contain enough information, clearly state that the policy does not specify."
        )
        log_entry = (
            f"🤖 [Generate Node - Attempt {retries + 1}] Generated context-grounded candidate answer "
            f"via Groq ({groq_model})."
        )

    user_prompt = f"POLICY CONTEXT:\n{context_str}\n\nQUESTION: {state['question']}"

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    candidate_answer = str(response.content).strip()

    log_step("   ↳ Draft Output:", f"'{candidate_answer[:120]}...'", "dim")
    current_log.append(log_entry)

    return {
        "candidate_answer": candidate_answer,
        "audit_log": current_log,
    }


def audit_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Audits the candidate answer using Gemini 2.5 Flash + Pydantic Structured Output.

    Acts as an adversarial 'LLM-as-a-Judge', inspecting every assertion against ground truth.
    """
    _check_api_keys()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "langchain-google-genai is required. Run 'pip install langchain-google-genai'."
        ) from exc

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    retries = state.get("retries", 0)

    log_step(
        "⚖️  [Node: Audit]",
        f"(Attempt {retries + 1}) Critic reviewing candidate answer via Google Gemini ({gemini_model})...",
        "bold yellow",
    )

    llm = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Enforce Pydantic v2 structured output
    auditor = llm.with_structured_output(AuditResult)

    context_str = "\n\n---\n\n".join(state["retrieved_docs"])

    audit_prompt = (
        "You are an uncompromising Compliance and Fact-Checking Auditor (LLM-as-a-Judge) for ApexPay Global.\n"
        "Your duty is to rigorously cross-examine the CANDIDATE ANSWER against the GROUND-TRUTH POLICY CONTEXT.\n\n"
        "EVALUATION CRITERIA:\n"
        "1. is_grounded: Set to True ONLY IF every single assertion, dollar amount, fee, timeline, percentage, "
        "and error code in the CANDIDATE ANSWER is explicitly verified by the GROUND-TRUTH CONTEXT. "
        "Set to False if ANY statement is fabricated, distorted, contradicted, or unsupported.\n"
        "2. confidence_score: Assign a float between 0.0 and 1.0 representing your certainty of grounding.\n"
        "3. hallucination_notes: If is_grounded is False, identify the exact ungrounded or fabricated claims. "
        "If is_grounded is True, confirm all statements match ground truth.\n"
        "4. correction_instructions: If is_grounded is False, provide exact instructions on what to fix, remove, "
        "or replace using only verified facts from the context. If is_grounded is True, set to empty string ''.\n\n"
        f"GROUND-TRUTH POLICY CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION:\n{state['question']}\n\n"
        f"CANDIDATE ANSWER TO AUDIT:\n{state['candidate_answer']}"
    )

    audit_result: AuditResult = auditor.invoke(audit_prompt)

    current_log = list(state.get("audit_log", []))

    if audit_result.is_grounded:
        verdict_summary = (
            f"✅ [Audit Node - Attempt {retries + 1}] PASSED: Answer is factually grounded. "
            f"Confidence: {audit_result.confidence_score:.2f}."
        )
        log_step("   ↳ Auditor Verdict:", "PASSED (Grounded = True) ✅", "bold green")
        new_retries = retries
    else:
        verdict_summary = (
            f"🚨 [Audit Node - Attempt {retries + 1}] FAILED: Hallucination detected! "
            f"Confidence: {audit_result.confidence_score:.2f}. Notes: {audit_result.hallucination_notes}"
        )
        log_step("   ↳ Auditor Verdict:", "FAILED (Hallucination Detected) 🚨", "bold red")
        log_step("   ↳ Hallucination Notes:", audit_result.hallucination_notes, "red")
        log_step("   ↳ Instructions:", audit_result.correction_instructions, "yellow")
        new_retries = retries + 1

    current_log.append(verdict_summary)

    return {
        "is_grounded": audit_result.is_grounded,
        "confidence_score": audit_result.confidence_score,
        "hallucination_notes": audit_result.hallucination_notes,
        "correction_instructions": audit_result.correction_instructions,
        "retries": new_retries,
        "audit_log": current_log,
    }


def decide_next_step(state: AgentState) -> str:
    """Conditional Edge: Determines whether to route to END or loop back to generate."""
    is_grounded = state.get("is_grounded", False)
    retries = state.get("retries", 0)

    if is_grounded:
        log_step("🏁 [Router]", "Answer is fully grounded. Routing to END.", "bold green")
        return "end"

    if retries <= 2:
        log_step(
            "🔁 [Router]",
            f"Factual discrepancy detected. Triggering self-correction loop (Retry {retries}/2) -> generate",
            "bold yellow",
        )
        return "generate"

    log_step("⚠️  [Router]", "Maximum retries (2) reached. Ending workflow.", "bold red")
    return "end"


def build_rag_auditor_graph():
    """Assembles and compiles the LangGraph StateGraph."""
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("audit", audit_node)

    # Wire Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "audit")

    # Add Conditional Routing
    workflow.add_conditional_edges(
        "audit",
        decide_next_step,
        {
            "generate": "generate",
            "end": END,
        },
    )

    return workflow.compile()


# Singleton compiled graph instance
_COMPILED_GRAPH = None


def get_agent_graph():
    """Lazily compiles and returns the LangGraph workflow graph."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_rag_auditor_graph()
    return _COMPILED_GRAPH


def run_agent(question: str, simulate_hallucination: bool = False) -> AgentResponse:
    """Executes the full RAG & Hallucination Auditor workflow for a user query.

    Args:
        question: The user's query about policies.
        simulate_hallucination: If True, forces initial generation to inject an error.

    Returns:
        AgentResponse: Validated output with answer, grounding flag, score, retries, and logs.
    """
    graph = get_agent_graph()

    initial_state: AgentState = {
        "question": question,
        "simulate_hallucination": simulate_hallucination,
        "retrieved_docs": [],
        "candidate_answer": "",
        "is_grounded": False,
        "confidence_score": 0.0,
        "hallucination_notes": "",
        "correction_instructions": "",
        "retries": 0,
        "audit_log": [f"🚀 [Workflow Initialized] Question: '{question}' | Simulate Hallucination: {simulate_hallucination}"],
    }

    log_step("\n==================================================", "", "bold blue")
    log_step("▶️  STARTING RAG AUDITOR WORKFLOW", f"Query: {question}", "bold blue")
    log_step("==================================================", "", "bold blue")

    final_state = graph.invoke(initial_state)

    retries_used = final_state.get("retries", 0)

    return AgentResponse(
        question=question,
        answer=final_state.get("candidate_answer", ""),
        is_grounded=final_state.get("is_grounded", False),
        confidence_score=final_state.get("confidence_score", 0.0),
        retries_used=retries_used,
        audit_log=final_state.get("audit_log", []),
        sources=final_state.get("retrieved_docs", []),
    )


if __name__ == "__main__":
    test_query = "What is the standard settlement timeline and the fee for express same-day settlement?"
    print(f"Running test agent with query: {test_query}\n")
    try:
        res = run_agent(test_query, simulate_hallucination=True)
        print("\n\nFINAL RESPONSE:")
        print(f"Answer: {res.answer}")
        print(f"Grounded: {res.is_grounded}")
        print(f"Confidence: {res.confidence_score}")
        print(f"Retries Used: {res.retries_used}")
        print(f"Audit Log Steps: {len(res.audit_log)}")
    except Exception as e:
        print(f"Execution notice: {e}")
