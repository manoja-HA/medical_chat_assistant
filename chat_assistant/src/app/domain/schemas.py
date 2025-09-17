"""
Request/response schemas (DTOs) for the Medical Quiz Assistant.
Transport layer schemas separate from domain models.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class QuizRequest(BaseModel):
    """Request schema for /quiz-me endpoint."""

    session_id: str = Field(
        ..., description="User session identifier", default_factory=lambda: str(uuid.uuid4())
    )
    topic: Optional[str] = Field(
        None,
        description="Filter by topic",
        examples=[
            "cardiology",
            "neurology",
            "orthopedics",
            "neonatology",
            "gastroenterology",
        ],
    )
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        None, description="Filter by difficulty"
    )
    exclude_seen: bool = Field(True, description="Exclude previously seen questions")


# --- New Quiz Me models ---
class QuizMeRequest(BaseModel):
    """Request for the simplified 'Quiz Me' flow (no session/state)."""

    command: Optional[str] = Field(
        default="Quiz Me",
        description="Command trigger. Defaults to 'Quiz Me'",
        examples=["Quiz Me"],
    )
    page_size: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of random questions to return (1-20)",
        examples=[1, 5],
    )


class QuizQuestion(BaseModel):
    """A single multiple-choice question with options and metadata."""

    question_id: str
    stem: str
    options: dict[str, str]
    topic: Optional[str] = None
    difficulty: Optional[str] = None


class QuizMeResponse(BaseModel):
    """Response for the 'Quiz Me' flow."""

    questions: list[QuizQuestion]
    version: dict[str, Any] = Field(
        ..., description="Version info: retriever, embed_model, model, build (YYYY-MM-DD)"
    )


class QuizResponse(BaseModel):
    """Response schema for /quiz-me endpoint."""

    question_id: str = Field(..., description="Question identifier")
    stem: str = Field(..., description="Question text")
    options: list[str] = Field(..., min_length=4, max_length=4, description="Answer options")
    metadata: dict[str, str] = Field(
        ..., description="Question metadata (subject, topic, difficulty)"
    )


class AnswerRequest(BaseModel):
    """Request schema for /answer endpoint."""

    question_id: str = Field(..., description="Question being answered")
    selected_option: Literal["A", "B", "C", "D"] = Field(
        ..., description="Selected option label (A-D)"
    )


class AnswerResponse(BaseModel):
    """Minimal response schema for /answer endpoint (only correctness)."""

    correct: bool = Field(..., description="Whether the answer is correct")
    correct_option: Literal["A", "B", "C", "D"] = Field(
        ..., description="The correct option label (A-D)"
    )


class ExplainRequest(BaseModel):
    """Request schema for /explain endpoint."""

    question_id: str = Field(..., description="Question to explain")
    mode: Literal["dataset", "rag"] = Field("dataset", description="Explanation mode")


class ExplainResponse(BaseModel):
    """Response schema for /explain endpoint."""

    reason_correct: str = Field(..., description="Why the correct answer is right")
    reason_incorrect: list[dict[str, str]] = Field(
        ..., description="Why incorrect options are wrong"
    )
    key_points: list[str] = Field(..., description="Key learning points")
    citations: list[str] = Field(..., description="Source citations (empty for dataset mode)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    source: Literal["dataset", "rag"] = Field(..., description="Source of explanation")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")


class IngestRequest(BaseModel):
    """Request schema for /ingest endpoint."""

    split: Literal["train", "validation", "test"] = Field(
        "train", description="Dataset split to ingest"
    )
    limit: Optional[int] = Field(None, description="Maximum number of questions to ingest")
    batch_size: int = Field(100, description="Batch size for processing")


class IngestResponse(BaseModel):
    """Response schema for /ingest endpoint."""

    questions_processed: int = Field(..., description="Number of questions processed")
    questions_skipped: int = Field(..., description="Number of questions skipped (already exists)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    dataset_version: str = Field(..., description="Dataset version hash")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: Literal["healthy", "unhealthy", "degraded"] = Field(
        ..., description="Service health status"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Health check timestamp"
    )
    services: dict[str, bool] = Field(..., description="Individual service health status")
    version: str = Field(..., description="Application version")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional health details")


class ReadinessResponse(BaseModel):
    """Readiness check response schema."""

    ready: bool = Field(..., description="Whether service is ready")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Readiness check timestamp"
    )
    dependencies: dict[str, dict[str, Any]] = Field(..., description="Dependency status details")
    version: str = Field(..., description="Application version")


class MetricsResponse(BaseModel):
    """Metrics response schema."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Metrics timestamp"
    )
    metrics: dict[str, Any] = Field(..., description="System metrics")


class RAGExplainOutput(BaseModel):
    """RAG explanation output schema for LLM responses."""

    reason_correct: str = Field(..., description="Explanation for correct answer")
    reason_incorrect: list[dict[str, str]] = Field(
        ..., description="Explanations for incorrect options"
    )
    key_points: list[str] = Field(..., description="Key learning points")
    citations: list[str] = Field(..., description="Source document citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
