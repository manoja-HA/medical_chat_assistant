"""
Domain ports (interfaces) for the Medical Quiz Assistant.
Implements dependency inversion principle per ADR requirements.
"""

from abc import abstractmethod
from typing import Any, Optional, Protocol

from .models import Question


class QuestionRepository(Protocol):
    """Port for question data access."""

    @abstractmethod
    async def get_random(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        exclude_ids: Optional[list[str]] = None,
    ) -> Optional[Question]:
        """Get a random question with optional filters."""
        pass

    @abstractmethod
    async def get_by_id(self, question_id: str) -> Optional[Question]:
        """Get question by ID."""
        pass

    @abstractmethod
    async def save(self, question: Question) -> None:
        """Save a question."""
        pass

    @abstractmethod
    async def get_by_topic(self, topic: str, limit: int = 10) -> list[Question]:
        """Get questions by topic."""
        pass


class Retriever(Protocol):
    """Port for document retrieval."""

    @abstractmethod
    async def retrieve(
        self, query: str, k: int = 5, topic: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents for query."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if retriever is healthy."""
        pass


class Reranker(Protocol):
    """Port for document reranking (optional)."""

    @abstractmethod
    async def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Rerank documents based on query relevance."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if reranker is healthy."""
        pass


class LLMService(Protocol):
    """Port for LLM inference."""

    @abstractmethod
    async def explain(
        self,
        query: str,
        contexts: list[str],
        question_stem: str,
        options: list[str],
        correct_idx: int,
    ) -> dict[str, Any]:
        """Generate explanation using LLM."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if LLM service is healthy."""
        pass

    @abstractmethod
    async def warm_up(self) -> bool:
        """Warm up the LLM service."""
        pass
