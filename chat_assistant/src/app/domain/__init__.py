"""
Domain layer for Medical Quiz Assistant.
Contains ports (interfaces), models (entities), and schemas (DTOs).
"""

from .models import ExplanationResult, HealthStatus, Question, RAGContext
from .ports import LLMService, QuestionRepository, Reranker, Retriever
from .schemas import (
    AnswerRequest,
    AnswerResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QuizRequest,
    QuizResponse,
    ReadinessResponse,
)

__all__ = [
    # Models (domain entities)
    "Question",
    "RAGContext",
    "ExplanationResult",
    "HealthStatus",
    # Ports (interfaces)
    "QuestionRepository",
    "Retriever",
    "Reranker",
    "LLMService",
    # Schemas (DTOs)
    "QuizRequest",
    "QuizResponse",
    "AnswerRequest",
    "AnswerResponse",
    "ExplainRequest",
    "ExplainResponse",
    "IngestRequest",
    "IngestResponse",
    "HealthResponse",
    "ReadinessResponse",
    "MetricsResponse",
]
