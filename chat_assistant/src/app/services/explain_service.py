"""
Explanation service for providing detailed explanations.
Supports both dataset-first and RAG-enhanced explanations per ADR-001.
"""

import logging
from typing import Any, Optional

from app.core.config import Settings
from app.domain.models import Question
from app.domain.ports import QuestionRepository
from app.domain.schemas import ExplainRequest, ExplainResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class ExplainService:
    """Service for providing detailed explanations with dataset-first approach."""

    def __init__(
        self,
        question_repo: QuestionRepository,
        rag_service: Optional[RAGService],
        settings: Settings,
    ):
        self.question_repo = question_repo
        self.rag_service = rag_service
        self.settings = settings

    async def get_explanation(self, request: ExplainRequest) -> ExplainResponse:
        """
        Get detailed explanation for a question using clean architecture.

        This method now uses only the repository pattern - no direct ChromaDB access.
        """
        logger.info(
            f"Getting explanation for question {request.question_id}, mode {request.mode}"
        )

        try:

            # Get question using repository (no direct ChromaDB access)
            question = await self.question_repo.get_by_id(request.question_id)

            if not question:
                logger.warning(f"Question {request.question_id} not found in repository")
                raise ValueError(f"Question {request.question_id} not found")

            # Generate explanation based on mode
            if request.mode == "dataset":
                response = await self._get_dataset_explanation(question)
            elif request.mode == "rag" and self.rag_service:
                response = await self._get_rag_explanation(question)
            else:
                # Fallback to dataset if RAG not available
                logger.warning(
                    "RAG mode requested but service not available, falling back to dataset"
                )
                response = await self._get_dataset_explanation(question)

            logger.info(f"Explanation generated for question {request.question_id}")
            return response

        except ValueError as e:
            logger.warning(f"Explanation request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating explanation: {e}")
            raise RuntimeError("Failed to generate explanation") from e

    async def _get_dataset_explanation(self, question: Question) -> ExplainResponse:
        """
        Get explanation using dataset content only.

        Uses the domain model directly - no external service calls.
        """
        logger.debug(f"Generating dataset explanation for question {question.id}")

        try:
            # Build reason for correct answer from domain model
            reason_correct = question.expl_correct or "No explanation available in dataset."

            # Build reasons for incorrect options
            reason_incorrect = []
            labels = ["A", "B", "C", "D"]

            if question.expl_by_option:
                # Use per-option explanations if available
                for idx in range(len(question.options)):
                    if idx != question.correct_idx:
                        explanation = question.expl_by_option.get(idx, "No explanation available.")
                        reason_incorrect.append(
                            {
                                "option_idx": labels[idx],
                                "short_reason": str(explanation),
                            }
                        )
            else:
                # Generate basic explanations for incorrect options
                for idx in range(len(question.options)):
                    if idx != question.correct_idx:
                        reason_incorrect.append(
                            {
                                "option_idx": labels[idx],
                                "short_reason": "This option is incorrect.",
                            }
                        )

            # Extract key points from correct explanation
            key_points = self._extract_key_points(reason_correct)

            response = ExplainResponse(
                reason_correct=reason_correct,
                reason_incorrect=reason_incorrect,
                key_points=key_points,
                citations=[],
                confidence=1.0,  # Dataset explanations are always confident
                source="dataset",
            )

            logger.debug(f"Dataset explanation generated with {len(key_points)} key points")
            return response

        except Exception as e:
            logger.error(f"Error generating dataset explanation: {e}")
            raise RuntimeError("Failed to generate dataset explanation") from e

    async def _get_rag_explanation(self, question: Question) -> ExplainResponse:
        """
        Get explanation using RAG-enhanced approach.

        Uses the RAG service through dependency injection - no direct external calls.
        """
        logger.debug(f"Generating RAG explanation for question {question.id}")

        if not self.rag_service:
            logger.warning("RAG service not available, falling back to dataset")
            return await self._get_dataset_explanation(question)

        try:
            # Retrieve relevant context using RAG service
            context = await self.rag_service.retrieve_context(question)
            logger.debug(f"Retrieved {len(context)} context documents")

            # Generate RAG explanation
            rag_output = await self.rag_service.generate_explanation(question, context)

            # Normalize reason_incorrect to match ExplainResponse schema
            normalized_reason_incorrect: list[dict[str, str]] = []
            labels = ["A", "B", "C", "D"]

            for item in rag_output.reason_incorrect:
                idx = item.get("option_idx")
                # Convert numeric index to label if needed
                try:
                    idx_int = int(idx)
                    label = labels[idx_int] if 0 <= idx_int < 4 else str(idx)
                except (ValueError, TypeError):
                    label = str(idx)

                normalized_reason_incorrect.append(
                    {
                        "option_idx": label,
                        "short_reason": str(item.get("short_reason", "This option is incorrect.")),
                    }
                )

            # Convert to response format
            response = ExplainResponse(
                reason_correct=rag_output.reason_correct,
                reason_incorrect=normalized_reason_incorrect,
                key_points=rag_output.key_points,
                citations=rag_output.citations,
                confidence=rag_output.confidence,
                source="rag",
            )

            logger.info(f"RAG explanation generated with confidence {rag_output.confidence}")
            return response

        except Exception as e:
            logger.error(f"RAG explanation failed: {e}, falling back to dataset")
            return await self._get_dataset_explanation(question)

    def _extract_key_points(self, explanation: str) -> list[str]:
        """
        Extract key learning points from explanation text.

        Simple extraction logic - can be enhanced later.
        """
        try:
            # Split by sentences and take first few meaningful ones
            sentences = explanation.split(". ")
            key_points = []

            for sentence in sentences[:3]:  # Take first 3 sentences
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Filter out very short sentences
                    # Clean up sentence
                    if not sentence.endswith("."):
                        sentence += "."
                    key_points.append(sentence)

            # Ensure we have at least one key point
            if not key_points and explanation:
                key_points.append(
                    explanation[:100] + "..." if len(explanation) > 100 else explanation
                )

            return key_points

        except Exception as e:
            logger.warning(f"Error extracting key points: {e}")
            return ["Key learning points could not be extracted."]

    async def get_explanation_quality_metrics(
        self, question_id: str, mode: str = "dataset"
    ) -> dict[str, Any]:
        """
        Get quality metrics for an explanation using clean architecture.

        Uses repository pattern instead of direct database access.
        """
        logger.info(f"Getting quality metrics for question {question_id}, mode {mode}")

        try:
            # Get question using repository
            question = await self.question_repo.get_by_id(question_id)
            if not question:
                raise ValueError(f"Question {question_id} not found")

            # Generate explanation
            if mode == "dataset":
                explanation = await self._get_dataset_explanation(question)
            else:
                explanation = await self._get_rag_explanation(question)

            # Calculate quality metrics
            metrics = {
                "explanation_length": len(explanation.reason_correct),
                "key_points_count": len(explanation.key_points),
                "citations_count": len(explanation.citations),
                "confidence": explanation.confidence,
                "source": explanation.source,
                "has_incorrect_explanations": len(explanation.reason_incorrect) > 0,
                "coverage_score": self._calculate_coverage_score(explanation, question),
            }

            logger.info(f"Quality metrics calculated for question {question_id}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            raise RuntimeError("Failed to calculate explanation quality metrics") from e

    def _calculate_coverage_score(self, explanation: ExplainResponse, question: Question) -> float:
        """
        Calculate how well the explanation covers the question content.

        Simple coverage calculation based on key terms overlap.
        """
        try:
            # Extract terms from question and explanation
            question_terms = set(question.stem.lower().split())
            explanation_terms = set(explanation.reason_correct.lower().split())

            # Remove common stop words
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "up",
                "about",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "among",
                "this",
                "that",
                "these",
                "those",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "can",
            }

            question_terms -= stop_words
            explanation_terms -= stop_words

            if not question_terms:
                return 1.0  # If no meaningful terms in question, consider fully covered

            # Calculate overlap
            overlap = len(question_terms.intersection(explanation_terms))
            coverage = overlap / len(question_terms)

            return min(coverage, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating coverage score: {e}")
            return 0.5  # Return neutral score on error
