"""FastAPI REST API server for the Multi-Model RAG & Hallucination Auditor.

Exposes:
- GET  /health: System health, vector DB status, and API key configurations.
- POST /ask: Executes the LangGraph RAG + Critic workflow and returns AgentResponse.
"""

import os
from contextlib import asynccontextmanager
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agent import run_agent
from src.ingest import ensure_vector_store
from src.schemas import AgentResponse, QueryRequest

# Load environment configuration
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes local vector store and verifies configuration on server startup."""
    print("🚀 [FastAPI] Initializing server and verifying local ChromaDB store...")
    try:
        ensure_vector_store()
        print("✅ [FastAPI] Vector store initialized successfully.")
    except Exception as exc:
        print(f"⚠️  [FastAPI] Notice during startup vector store check: {exc}")
    yield
    print("🛑 [FastAPI] Shutting down server.")


app = FastAPI(
    title="Multi-Model RAG & Hallucination Auditor Agent API",
    description=(
        "Autonomous RAG agent with LLM-as-a-Judge critic feedback loops. "
        "Powered by Groq (Llama 3.3 70B), Google Gemini 2.5 Flash, and local ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for flexible frontend/client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health Check and System Diagnostics", tags=["System"])
async def health_check() -> Dict[str, object]:
    """Returns operational status, model credentials verification, and ChromaDB health."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    groq_ready = bool(groq_key and not groq_key.startswith("your_"))
    google_ready = bool(google_key and not google_key.startswith("your_"))

    chroma_ready = False
    try:
        store = ensure_vector_store()
        chroma_ready = store._collection.count() > 0
    except Exception:
        chroma_ready = False

    # Operational as long as Google Gemini and ChromaDB are available
    is_healthy = bool(google_ready and chroma_ready)
    generator_info = (
        f"Groq ({os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')})"
        if groq_ready
        else f"Google Gemini ({os.getenv('GENERATOR_GEMINI_MODEL', 'gemini-1.5-flash')}) [Fallback]"
    )

    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": "Multi-Model RAG & Hallucination Auditor",
        "version": "1.0.0",
        "groq_configured": groq_ready,
        "google_configured": google_ready,
        "chroma_initialized": chroma_ready,
        "models": {
            "generator": generator_info,
            "auditor": os.getenv("GEMINI_MODEL", "gemini-2.5-flash / gemini-1.5-flash"),
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2 (Local CPU)",
        },
    }


@app.post(
    "/ask",
    response_model=AgentResponse,
    summary="Query Knowledge Base with Self-Correcting Auditor",
    tags=["Agent"],
)
async def ask_agent(request: QueryRequest) -> AgentResponse:
    """Executes the full Multi-Model RAG and Hallucination Auditor workflow.

    - Retrieves relevant context from local ChromaDB.
    - Generates candidate answer with Groq (Llama 3.3 70B).
    - Audits factual grounding with Google Gemini 2.5 Flash.
    - If hallucination detected, automatically loops back to self-correct up to 2 times.
    """
    try:
        response = run_agent(
            question=request.question,
            simulate_hallucination=request.simulate_hallucination,
        )
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agent workflow: {str(exc)}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    print(f"Starting FastAPI server on http://{host}:{port} ...")
    uvicorn.run("src.api:app", host=host, port=port, reload=True)
