"""
Quiz service for selecting random questions without session management.
Stateless: provides a 'quiz_me' method that returns random questions.
"""

import logging
from datetime import datetime

from app.core.config import Settings
from app.domain.ports import QuestionRepository
from app.domain.schemas import (
    QuizMeRequest,
    QuizMeResponse,
    QuizQuestion,
)

logger = logging.getLogger(__name__)


class QuizService:
    """Stateless quiz service for random question selection."""

    def __init__(self, question_repo: QuestionRepository, settings: Settings):
        self.question_repo = question_repo
        self.settings = settings

    async def quiz_me(self, request: QuizMeRequest) -> QuizMeResponse:
        """
        Get random questions using repository pattern (no direct ChromaDB access).

        This method now properly uses the injected question_repo instead of
        bypassing it with direct ChromaDB calls.
        """
        logger.info(f"Quiz Me requested: command={request.command}, page_size={request.page_size}")

        questions: list[QuizQuestion] = []

        try:
            # Use repository to get random questions (this replaces the ChromaDB bypass)
            repo_questions = await self.question_repo.get_random_questions(
                count=min(request.page_size, 20), filters=None  # No filters for basic quiz me
            )

            if not repo_questions:
                logger.warning("No questions available from repository")
                return QuizMeResponse(
                    questions=[],
                    version={
                        "retriever": "repository",
                        "embed_model": "none",
                        "model": "dataset-only",
                        "build": datetime.utcnow().strftime("%Y-%m-%d"),
                    },
                )

            # Convert repository questions to response format
            for q in repo_questions:
                opts_dict = {
                    "A": q.options[0] if len(q.options) > 0 else "",
                    "B": q.options[1] if len(q.options) > 1 else "",
                    "C": q.options[2] if len(q.options) > 2 else "",
                    "D": q.options[3] if len(q.options) > 3 else "",
                }

                questions.append(
                    QuizQuestion(
                        question_id=q.id,
                        stem=q.stem,
                        options=opts_dict,
                        topic=q.topic,
                        difficulty=q.difficulty,
                    )
                )

            logger.info(f"Retrieved {len(questions)} questions from repository")

        except Exception as e:
            logger.error(f"Error retrieving questions from repository: {e}")
            # Return empty result instead of trying ChromaDB fallback
            return QuizMeResponse(
                questions=[],
                version={
                    "retriever": "error",
                    "embed_model": "none",
                    "model": "dataset-only",
                    "build": datetime.utcnow().strftime("%Y-%m-%d"),
                    "error": str(e),
                },
            )

        return QuizMeResponse(
            questions=questions,
            version={
                "retriever": "repository",
                "embed_model": self.settings.embedding_model,
                "model": "dataset-only",
                "build": datetime.utcnow().strftime("%Y-%m-%d"),
            },
        )
