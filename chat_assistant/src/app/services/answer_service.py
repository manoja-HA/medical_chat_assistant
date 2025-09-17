import logging

from app.domain.ports import QuestionRepository
from app.domain.schemas import AnswerRequest, AnswerResponse

logger = logging.getLogger(__name__)


class AnswerService:
    """Stateless service for processing quiz answers (correctness only)."""

    def __init__(self, question_repo: QuestionRepository):
        self.question_repo = question_repo

    async def process_answer(self, request: AnswerRequest) -> AnswerResponse:
        """
        Process submitted answer and return correctness only.
        Uses repository to load question data.
        """
        logger.info(
            f"Processing answer for question {request.question_id}"
        )

        try:
            question = await self.question_repo.get_by_id(request.question_id)

            if not question:
                logger.warning(f"Question {request.question_id} not found")
                raise ValueError(f"Question {request.question_id} not found")

            selected_idx = self._get_selected_index(request)

            if selected_idx < 0 or selected_idx >= len(question.options):
                raise ValueError(f"Invalid option index: {selected_idx}")

            is_correct = selected_idx == question.correct_idx

            # Map correct index to option label
            idx_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
            correct_option = idx_to_label.get(question.correct_idx, "A")

            response = AnswerResponse(correct=is_correct, correct_option=correct_option)

            logger.info(f"Answer processed: correct={is_correct}, correct_option={correct_option}")
            return response

        except ValueError as e:
            logger.warning(f"Answer processing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing answer: {e}")
            raise RuntimeError("Failed to process answer") from e

    def _get_selected_index(self, request: AnswerRequest) -> int:
        """Determine selected option index from request.selected_option (A-D)."""
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        if request.selected_option in label_map:
            return label_map[request.selected_option]
        raise ValueError("No valid selection provided")
