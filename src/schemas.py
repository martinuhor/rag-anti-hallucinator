"""Data models and schemas for the Multi-Model RAG & Hallucination Auditor Agent.

Uses Pydantic v2 for robust validation and structured LLM outputs.
"""

from typing import List
from pydantic import BaseModel, Field


class AuditResult(BaseModel):
    """Structured evaluation output returned by the Critic / Auditor LLM (Gemini 2.5 Flash).

    The critic assesses whether all claims in the candidate answer are strictly
    supported by the retrieved ground-truth context.
    """

    is_grounded: bool = Field(
        ...,
        description=(
            "Set to True ONLY if every single claim, number, fee, deadline, and policy in "
            "the candidate answer is directly substantiated by the provided context. "
            "Set to False if there are any fabrications, distortions, or ungrounded claims."
        ),
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score between 0.0 and 1.0 representing the auditor's certainty "
            "regarding factual grounding and compliance."
        ),
    )
    hallucination_notes: str = Field(
        ...,
        description=(
            "Detailed critique analyzing any discrepancies, fabricated figures, or ungrounded claims. "
            "If fully grounded, explicitly confirm that all assertions match the retrieved context."
        ),
    )
    correction_instructions: str = Field(
        ...,
        description=(
            "Clear, actionable guidance instructing the generator LLM on what specific "
            "claims to remove, replace, or correct using only the ground truth. "
            "Leave as an empty string if is_grounded is True."
        ),
    )


class QueryRequest(BaseModel):
    """User input payload for querying the RAG auditor workflow."""

    question: str = Field(
        ...,
        min_length=3,
        description="User question regarding fintech settlement policies or compliance rules.",
        examples=["What is the standard merchant settlement timeline and the fee for express settlement?"],
    )
    simulate_hallucination: bool = Field(
        default=False,
        description=(
            "Demonstration flag: when True, instructs the generator to intentionally inject "
            "a plausible factual error on Attempt 1. This triggers the Gemini Critic node to catch "
            "the hallucination and force an automated self-correction loop."
        ),
    )


class AgentResponse(BaseModel):
    """Final verified response returned by the multi-model RAG workflow."""

    question: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="The final, auditor-verified answer.")
    is_grounded: bool = Field(..., description="True if final answer is verified as grounded.")
    confidence_score: float = Field(..., description="Confidence score from the auditor node.")
    retries_used: int = Field(
        ...,
        ge=0,
        description="Number of self-correction feedback iterations executed (0 = passed first attempt).",
    )
    audit_log: List[str] = Field(
        ...,
        description="Chronological log tracing node transitions, attempts, and auditor critiques.",
    )
    sources: List[str] = Field(
        ...,
        description="Excerpts of retrieved context chunks used to answer the question.",
    )
