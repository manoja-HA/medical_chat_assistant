"""
Domain models for the Medical Quiz Assistant.
Core business entities separate from transport DTOs.
"""

from datetime import UTC, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Question(BaseModel):
    """Core question domain model."""

    id: str = Field(..., description="Unique question identifier")
    stem: str = Field(..., description="Question text/prompt")
    options: list[str] = Field(
        ..., min_length=4, max_length=4, description="Exactly 4 multiple choice options"
    )
    correct_idx: int = Field(..., ge=0, le=3, description="Index of correct option (0-3)")
    expl_correct: Optional[str] = Field(None, description="Explanation for correct answer")
    expl_by_option: Optional[dict[int, str]] = Field(None, description="Per-option explanations")
    subject: Optional[str] = Field(None, description="Medical subject area")
    topic: Optional[str] = Field(None, description="Specific topic within subject")
    difficulty: Optional[str] = Field(None, description="Question difficulty level")
    split: Literal["train", "validation", "test"] = Field(..., description="Dataset split")

    def get_metadata(self) -> dict[str, str]:
        """Get question metadata as string dict."""
        return {
            "subject": self.subject or "",
            "topic": self.topic or "",
            "difficulty": self.difficulty or "",
            "split": self.split,
        }


class SessionState(BaseModel):
    """User session state domain model."""

    session_id: str = Field(..., description="Session identifier")
    current_question_id: Optional[str] = Field(None, description="Currently active question")
    seen_questions: list[str] = Field(
        default_factory=list, description="Previously seen question IDs"
    )
    topic: Optional[str] = Field(None, description="Session topic filter")
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        None, description="Session difficulty filter"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Session creation time"
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last activity timestamp"
    )


class QuizAttempt(BaseModel):
    """Individual quiz attempt domain model."""

    session_id: str = Field(..., description="Session identifier")
    question_id: str = Field(..., description="Question answered")
    selected_idx: int = Field(..., ge=0, le=3, description="Selected option")
    correct: bool = Field(..., description="Whether answer was correct")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Attempt timestamp"
    )
    response_time_ms: Optional[int] = Field(
        None, description="Time taken to answer in milliseconds"
    )


class RAGContext(BaseModel):
    """RAG context document."""

    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Document source")
    score: float = Field(..., description="Relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExplanationResult(BaseModel):
    """Explanation generation result."""

    reason_correct: str = Field(..., description="Why the correct answer is right")
    reason_incorrect: list[dict[str, str]] = Field(
        ..., description="Why incorrect options are wrong"
    )
    key_points: list[str] = Field(..., description="Key learning points")
    citations: list[str] = Field(..., description="Source citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    source: Literal["dataset", "rag"] = Field(..., description="Source of explanation")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")


class HealthStatus(BaseModel):
    """System health status."""

    status: Literal["healthy", "unhealthy", "degraded"] = Field(
        ..., description="Overall health status"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Health check timestamp"
    )
    services: dict[str, bool] = Field(..., description="Individual service health status")
    version: str = Field(..., description="Application version")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional health details")
