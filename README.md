# 🛡️ Multi-Model RAG & Hallucination Auditor Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq Llama-3.3-70B](https://img.shields.io/badge/Generator-Groq%20Llama%203.3%2070B-f55036.svg)](https://groq.com)
[![Google Gemini-2.5-Flash](https://img.shields.io/badge/Auditor-Gemini%202.5%20Flash-4285F4.svg)](https://aistudio.google.com)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB%20(Local)-green.svg)](https://www.trychroma.com)
[![Cost](https://img.shields.io/badge/Cost-100%25%20Free%20Tier-brightgreen.svg)]()

An autonomous, production-ready, self-correcting RAG pipeline that answers domain-specific queries from a local knowledge base and applies an automated **"LLM-as-a-Judge"** critic node to audit factual grounding, detect subtle hallucinations, and trigger iterative self-correction loops before returning verified answers.

---

## 📌 Architectural Philosophy: Why Multi-Model Routing?

Most standard RAG pipelines deploy a single model to both generate answers and evaluate its own work. In production, this causes critical failure modes:
1. **Model Sycophancy & Confirmation Bias:** LLMs are conditioned to agree with their prior generation. When a model re-reads its own hallucinated output, it shares the same internal token probability biases and rarely flags subtle fabrications.
2. **Heterogeneous Adversarial Verification:** This system decouples the **Generator** from the **Auditor**:
   - **Generator (Meta Llama 3.3 70B via Groq LPUs):** Delivers ultra-low latency inference (~300 tokens/sec) for deep context synthesis.
   - **Auditor / Critic (Google Gemini 2.5 Flash via native Pydantic v2 structured outputs):** An independent model family acting as an adversarial compliance judge. It cross-examines every figure, date, fee, and claim strictly against the retrieved ground truth.
3. **100% Zero-Cost Local Vector Storage:** Uses `chromadb` paired with `sentence-transformers/all-MiniLM-L6-v2` running entirely on your local CPU—zero external embedding API costs and complete data privacy.

---

## 🔄 LangGraph Architecture Workflow

```
                        ┌──────────────────────────────┐
                        │          User Query          │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │      [Node 1: Retrieve]      │
                        │   ChromaDB + all-MiniLM-L6   │
                        │   Top-3 Semantic Chunks      │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │      [Node 2: Generate]      │◄─────────────────────────┐
                        │   Groq Llama 3.3 70B         │                          │
                        │   Synthesizes Candidate      │                          │
                        └──────────────┬───────────────┘                          │
                                       │                                          │
                                       ▼                                          │
                        ┌──────────────────────────────┐                          │
                        │       [Node 3: Audit]        │                          │
                        │   Gemini 2.5 Flash Critic    │                          │
                        │   Pydantic Structured Output │                          │
                        └──────────────┬───────────────┘                          │
                                       │                                          │
                                       ▼                                          │
                            /──────────────────────\                              │
                           /   Is Answer Grounded?  \                             │
                           \  and Retries < 2?      /                             │
                            \──────────────────────/                              │
                                   /        \                                     │
                        [No: Fail]          [Yes: Pass or Max Retries]            │
                                 /            \                                   │
                                ▼              ▼                                  │
                    ┌─────────────────┐  ┌──────────────────────────────────┐     │
                    │ Route Feedback  │  │               END                │     │
                    │ Instructions    │  │ Verified AgentResponse in JSON   │     │
                    └────────┬────────┘  └──────────────────────────────────┘     │
                             │                                                    │
                             └────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
rag-anti-hallucinator-project/
├── .env.example             # Template for free Groq and Gemini API keys
├── requirements.txt         # Pinned, tested dependencies
├── README.md                # Comprehensive documentation
├── ui.py                    # Interactive Streamlit dashboard with hallucination toggle
├── data/
│   └── sample_policy.md     # Dense fintech compliance policy (settlements, KYC, error codes)
└── src/
    ├── __init__.py
    ├── schemas.py           # Pydantic v2 schemas: AuditResult, QueryRequest, AgentResponse
    ├── ingest.py            # Local HuggingFace embedding generator & ChromaDB auto-indexer
    ├── agent.py             # LangGraph StateGraph (retrieve -> generate -> audit -> feedback loop)
    └── api.py               # FastAPI server (GET /health, POST /ask)
```

---

## ⚡ Quickstart Setup

### 1. Set Up Virtual Environment

```bash
# Navigate to project directory
cd "/Users/mac/Desktop/RAG anti hallucinator project "

# Create virtual environment with Python 3.11+
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Free API Keys

Obtain free-tier keys (no credit card required):
- **Groq API Key:** [https://console.groq.com](https://console.groq.com)
- **Google Gemini API Key:** [https://aistudio.google.com](https://aistudio.google.com)

Create your `.env` file:
```bash
cp .env.example .env
```

Open `.env` and insert your credentials:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GOOGLE_API_KEY=AIzaSy_your_gemini_api_key_here
```

### 3. Initialize Local Knowledge Base (ChromaDB)

Run the standalone ingestion script to parse and index `data/sample_policy.md`:
```bash
python src/ingest.py
```
*(Note: If you skip this, the auto-ingest helper will automatically build the database on first run).*

---

## 🚀 Running the Applications

### Option A: Launch Interactive Streamlit UI (Recommended for Demos)

```bash
streamlit run ui.py
```
Open **`http://localhost:8501`** in your browser.
- Select from realistic fintech questions or enter your own.
- Toggle **"Simulate Hallucination (Test Self-Correction)"** to watch the Critic reject a fabricated claim and auto-correct it in real time!
- Inspect the step-by-step audit trail, confidence score, and source documents.

---

### Option B: Run FastAPI REST API Server

```bash
python src/api.py
# Or via uvicorn directly:
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation will be available at: **`http://localhost:8000/docs`**.

#### API Testing via `curl`:

**Health Check:**
```bash
curl -X GET http://localhost:8000/health
```

**Query with Self-Correction Simulation:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the standard merchant settlement timeline and the fee for express settlement?",
    "simulate_hallucination": true
  }'
```

---

## 🔍 Example JSON Output: Hallucination Caught & Corrected

Below is an actual JSON output demonstrating Attempt 1 being flagged by the Gemini Critic for an altered fee, triggering an automated revision that succeeds on Attempt 2:

```json
{
  "question": "What is the standard merchant settlement timeline and the fee for express settlement?",
  "answer": "Under ApexPay Global policy, the standard merchant settlement timeline (T+2) finalizes payout batches two business days post-transaction settlement date at 17:00 UTC with no surcharge. Express Same-Day Settlement (T+0) is available for approved merchants with over 6 months of processing history for transactions processed before 11:00 UTC, subject to a flat 1.2% surcharge on the gross batch volume.",
  "is_grounded": true,
  "confidence_score": 0.98,
  "retries_used": 1,
  "audit_log": [
    "🚀 [Workflow Initialized] Question: 'What is the standard merchant settlement timeline and the fee for express settlement?' | Simulate Hallucination: True",
    "🔍 [Retrieve Node] Retrieved 3 policy chunks from local ChromaDB.",
    "🤖 [Generate Node - Attempt 1] Generated draft with intentional hallucination via Groq (llama-3.3-70b-versatile) for demonstration.",
    "🚨 [Audit Node - Attempt 1] FAILED: Hallucination detected! Confidence: 0.20. Notes: The candidate answer states that Express Same-Day Settlement carries a 3.5% fee and requires a $500 monthly subscription. According to Section 1.1 of the policy, express settlement incurs a flat 1.2% surcharge with no monthly subscription requirement.",
    "🔄 [Generate Node - Attempt 2] Regenerated revised answer via Groq incorporating auditor critique.",
    "✅ [Audit Node - Attempt 2] PASSED: Answer is factually grounded. Confidence: 0.98."
  ],
  "sources": [
    "### 1.1 Settlement Cadence\n- **Standard Settlement (T+2):** Merchant payout batches are finalized two business days post-transaction settlement date at 17:00 UTC with no surcharge.\n- **Express Same-Day Settlement (T+0):** Available for approved merchants with over 6 months of processing history. Transactions processed before 11:00 UTC settle on the same calendar day for a flat 1.2% surcharge on the gross batch volume.\n- **Weekend & Holiday Processing:** Standard settlements do not execute on Federal Reserve or TARGET2 bank holidays; funds roll over to the subsequent business day at 09:00 UTC.",
    "### 1.2 Daily Volume Ceilings & Reserves\n- **Tier 1 Merchant Cap:** Maximum daily aggregate settlement volume is capped at $250,000 USD equivalent.\n- **Tier 2 Merchant Cap:** Maximum daily aggregate settlement volume is capped at $50,000 USD equivalent.\n- **High-Risk Rolling Reserve:** Merchants assigned high-risk Merchant Category Codes (MCC 7995 for gaming/betting, MCC 5967 for direct marketing) are subject to a mandatory 5% rolling reserve withheld from each batch payout. Reserve funds are held in escrow for exactly 90 calendar days before automatic release to the merchant's operating account."
  ]
}
```

---

## 🧪 Knowledge Base Test Queries

The included `data/sample_policy.md` contains strict parameters for testing:
1. **Settlement Schedules:** Standard T+2 vs Express T+0 (1.2% fee, 11:00 UTC cutoff).
2. **Dispute Rules:** 30 calendar days pre-arbitration window, 14 days evidence deadline, $15 non-refundable fee.
3. **Surveillance Thresholds:** 0.90% chargeback ratio triggers 60-day probationary surveillance; >1.20% triggers 10% liquidity holdback.
4. **KYC Tiers:** Tier 1 ($1,000 lifetime limit), Tier 2 ($25,000 monthly limit), Tier 3 (Unlimited with UBO >25% declaration).
5. **API Error Codes:** `ERR_FX_SLIP_409` (0.35% slippage, 60s window), `ERR_SANCTION_BLOCK_403` (freeze & 4h compliance alert), `ERR_SETTLE_CAP_422` ($100k SEPA, $50k ACH), `ERR_VELOCITY_BREACH_429` (500 req/min).

---

## 🛡️ Production Standards
- **Strict Typing:** All data models enforced via Pydantic v2.
- **Fail-Safe Ingestion:** If `./chroma_db` is missing or deleted, the app automatically runs `ensure_vector_store()` without throwing unhandled exceptions.
- **Error Guardrails:** Informative diagnostic error messages if API keys are missing or invalid.
- **Auditor Independence:** Zero shared state or temperature leakage between Generator and Critic.

---

## 📜 License
MIT License. Built for production-grade agentic RAG benchmarking.
