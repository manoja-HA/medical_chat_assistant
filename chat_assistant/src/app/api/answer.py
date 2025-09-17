"""
Answer API endpoints for Medical Quiz Assistant.
Implements deterministic answer processing per ADR requirements.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.domain.schemas import AnswerRequest, AnswerResponse
from app.services.answer_service import AnswerService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_answer_service(settings: Settings = Depends(get_settings)) -> AnswerService:
    """Dependency to get answer service with ChromaDB repositories (stateless)."""
    from app.repositories.chroma_question_repository import ChromaQuestionRepository

    question_repo = ChromaQuestionRepository(settings)
    return AnswerService(question_repo)


@router.post("/submit", response_model=AnswerResponse)
async def submit_answer(
    request: AnswerRequest, answer_service: AnswerService = Depends(get_answer_service)
) -> AnswerResponse:
    """
    Submit an answer for a quiz question and get validation response.

    Accepts an answer from the user and responds with whether the answer is
    correct or incorrect, along with explanations.

    Implements deterministic answer checking with dataset-based explanations.
    """
    try:
        logger.info(
            f"Processing answer for question {request.question_id}, selected_option={request.selected_option}"
        )

        response = await answer_service.process_answer(request)

        logger.info(
            f"Answer processed: correct={response.correct}, correct_option={response.correct_option}"
        )
        return response

    except ValueError as e:
        logger.warning(f"Answer submission failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in answer endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
