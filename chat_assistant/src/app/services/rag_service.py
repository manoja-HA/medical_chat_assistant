"""
Fixed RAG service for enhanced explanations using retrieval-augmented generation.
Implements proper RAG pipeline with ChromaDB integration and Ollama LLM.
"""

import hashlib
import logging
from typing import Any

from app.core.config import Settings
from app.domain.models import Question
from app.domain.schemas import RAGExplainOutput
from app.repositories.chroma_vector_repository import ChromaVectorRepository
from app.services.cache_service import CacheService
from app.services.llm_service import OllamaLLMService

logger = logging.getLogger(__name__)


class RAGService:
    """Production RAG service with proper ChromaDB and LLM integration."""

    def __init__(
        self,
        vector_repo: ChromaVectorRepository,
        llm_service: OllamaLLMService,
        cache_service: CacheService,
        settings: Settings,
    ):
        """
        Initialize RAG service with properly injected dependencies.

        Args:
            vector_repo: ChromaDB vector repository for retrieval
            llm_service: Ollama LLM service for generation
            cache_service: Cache service for performance
            settings: Application settings
        """
        self.vector_repo = vector_repo
        self.llm_service = llm_service
        self.cache_service = cache_service
        self.settings = settings

        # RAG configuration
        self.top_k = settings.rag_top_k or 5
        self.confidence_threshold = settings.rag_confidence_threshold or 0.7
        self.max_context_length = 2000  # Characters

    async def retrieve_context(self, question: Question) -> list[dict[str, Any]]:
        """
        Retrieve relevant context for question explanation using ChromaDB.

        Args:
            question: Question to get context for

        Returns:
            List of relevant context documents with similarity scores
        """
        logger.info(f"Retrieving RAG context for question {question.id}")

        try:
            # Build comprehensive query from question content
            query_parts = [question.stem]

            # Add options for better context matching
            if question.options:
                query_parts.extend(question.options)

            # Add topic/subject for domain-specific retrieval
            if question.topic:
                query_parts.append(f"Topic: {question.topic}")
            if question.subject:
                query_parts.append(f"Subject: {question.subject}")

            query = " ".join(query_parts)

            # Check cache first
            cache_key = self._build_cache_key("context", question.id, query)
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                logger.debug(f"Retrieved cached context for question {question.id}")
                return cached_result

            # Retrieve from explanations collection using ChromaDB
            context_docs = await self.vector_repo.similarity_search(
                query=query,
                collection_name="explanations",
                k=self.top_k,
                filters=(
                    {
                        "subject": question.subject,
                        "topic": question.topic,
                    }
                    if question.subject or question.topic
                    else None
                ),
            )

            # Filter by relevance threshold
            relevant_docs = [
                doc
                for doc in context_docs
                if doc.get("similarity", 0) >= 0.3  # Minimum relevance threshold
            ]

            # Cache the result
            await self.cache_service.set(cache_key, relevant_docs, ttl=3600)

            logger.info(f"Retrieved {len(relevant_docs)} relevant context documents")
            return relevant_docs

        except Exception as e:
            logger.error(f"Error retrieving context for question {question.id}: {e}")
            return []  # Return empty context instead of failing

    async def generate_explanation(
        self, question: Question, context: list[dict[str, Any]]
    ) -> RAGExplainOutput:
        """
        Generate RAG-enhanced explanation using retrieved context and LLM.

        Args:
            question: Question to explain
            context: Retrieved context documents

        Returns:
            RAG explanation output with confidence score
        """
        logger.info(f"Generating RAG explanation for question {question.id}")

        try:
            # Check cache for complete explanation
            cache_key = self._build_cache_key("explanation", question.id, str(len(context)))
            cached_explanation = await self.cache_service.get(cache_key)
            if cached_explanation:
                logger.debug(f"Retrieved cached RAG explanation for question {question.id}")
                return cached_explanation

            # Prepare context for LLM
            formatted_context = self._format_context_for_llm(context)

            # Check if we have sufficient context
            if not context or len(formatted_context) < 50:
                logger.warning(
                    f"Insufficient context for RAG explanation of question {question.id}"
                )
                return self._create_fallback_explanation(question, "insufficient_context")

            # Generate explanation using LLM service
            explanation = await self.llm_service.generate_explanation(question, context)

            # Enhance with citation information
            enhanced_explanation = self._add_citations(explanation, context)

            # Cache the explanation
            await self.cache_service.set(cache_key, enhanced_explanation, ttl=7200)

            logger.info(
                f"Generated RAG explanation with confidence {enhanced_explanation.confidence}, "
                f"citations: {len(enhanced_explanation.citations)}"
            )
            return enhanced_explanation

        except Exception as e:
            logger.error(f"Error generating RAG explanation for question {question.id}: {e}")
            return self._create_fallback_explanation(question, "generation_error")

    def _format_context_for_llm(self, context: list[dict[str, Any]]) -> str:
        """Format retrieved context for LLM input."""
        if not context:
            return "No relevant context found."

        context_parts = []
        total_length = 0

        for i, doc in enumerate(context[: self.top_k]):
            content = doc.get("content", "").strip()
            similarity = doc.get("similarity", 0)

            if not content:
                continue

            # Add similarity-weighted context
            context_part = f"[Context {i+1}, Relevance: {similarity:.2f}]\n{content}\n"

            # Check length limits
            if total_length + len(context_part) > self.max_context_length:
                break

            context_parts.append(context_part)
            total_length += len(context_part)

        formatted = "\n".join(context_parts)
        logger.debug(f"Formatted {len(context_parts)} context documents ({total_length} chars)")
        return formatted

    def _add_citations(
        self, explanation: RAGExplainOutput, context: list[dict[str, Any]]
    ) -> RAGExplainOutput:
        """Add citation information to explanation."""
        try:
            # Extract source information from context
            citations = []
            for doc in context:
                doc_id = doc.get("id", "unknown")
                metadata = doc.get("metadata", {})

                # Create citation string
                citation = doc_id
                if metadata.get("question_id"):
                    citation = f"Question {metadata['question_id']}"
                if metadata.get("topic"):
                    citation += f" ({metadata['topic']})"

                citations.append(citation)

            # Update explanation with citations
            explanation.citations = citations[:3]  # Limit to top 3 citations
            return explanation

        except Exception as e:
            logger.error(f"Error adding citations: {e}")
            return explanation

    def _create_fallback_explanation(self, question: Question, reason: str) -> RAGExplainOutput:
        """Create fallback explanation - FIXED to exclude correct option from incorrect list."""
        logger.info(f"Creating fallback explanation for question {question.id}, reason: {reason}")

        # Use dataset explanation if available
        reason_correct = (
            question.expl_correct or "This is the correct answer based on medical knowledge."
        )

        # Generate incorrect explanations ONLY for wrong options
        reason_incorrect = []
        labels = ["A", "B", "C", "D"]

        for idx in range(len(question.options)):
            # CRITICAL: Only include options that are NOT correct
            if idx != question.correct_idx:
                reason_incorrect.append(
                    {
                        "option_idx": labels[idx],
                        "short_reason": f"Option {labels[idx]} is not the best answer for this medical question.",
                    }
                )

        # Generate key points
        key_points = []
        if question.topic:
            key_points.append(f"This question relates to {question.topic}")
        if question.subject:
            key_points.append(f"Understanding {question.subject} concepts is important")
        if not key_points:
            key_points.append("Review the relevant medical concepts for this question")

        return RAGExplainOutput(
            reason_correct=reason_correct,
            reason_incorrect=reason_incorrect,
            key_points=key_points,
            citations=[],
            confidence=0.5,
        )

    def _build_cache_key(self, operation: str, question_id: str, content: str) -> str:
        """Build consistent cache key for RAG operations."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"rag_{operation}_{question_id}_{content_hash}"

    async def get_retrieval_metrics(
        self, question: Question, context: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Get retrieval quality metrics for monitoring."""
        try:
            if not context:
                return {
                    "retrieval_count": 0,
                    "avg_similarity": 0.0,
                    "context_length": 0,
                    "coverage_score": 0.0,
                }

            # Calculate metrics
            similarities = [doc.get("similarity", 0) for doc in context]
            context_length = sum(len(doc.get("content", "")) for doc in context)

            # Simple coverage calculation
            question_terms = set(question.stem.lower().split())
            context_terms = set()
            for doc in context:
                content_terms = set(doc.get("content", "").lower().split())
                context_terms.update(content_terms)

            coverage_score = 0.0
            if question_terms:
                overlap = len(question_terms.intersection(context_terms))
                coverage_score = overlap / len(question_terms)

            return {
                "retrieval_count": len(context),
                "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
                "max_similarity": max(similarities) if similarities else 0.0,
                "context_length": context_length,
                "coverage_score": coverage_score,
            }

        except Exception as e:
            logger.error(f"Error calculating retrieval metrics: {e}")
            return {"error": str(e)}

    async def health_check(self) -> dict[str, Any]:
        """Check RAG service health."""
        try:
            # Check vector repository
            vector_health = await self.vector_repo.health_check()

            # Check LLM service
            llm_health = await self.llm_service.health_check()

            # Check cache service
            cache_health = await self.cache_service.health_check()

            overall_health = (
                vector_health
                and llm_health.get("status") == "healthy"
                and cache_health.get("status") == "healthy"
            )

            return {
                "status": "healthy" if overall_health else "degraded",
                "components": {
                    "vector_repository": vector_health,
                    "llm_service": llm_health,
                    "cache_service": cache_health,
                },
                "configuration": {
                    "top_k": self.top_k,
                    "confidence_threshold": self.confidence_threshold,
                    "max_context_length": self.max_context_length,
                },
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
