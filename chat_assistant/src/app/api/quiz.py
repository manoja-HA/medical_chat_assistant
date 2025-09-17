import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.domain.schemas import (
    QuizMeRequest,
    QuizMeResponse,
)
from app.services.quiz_service import QuizService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_quiz_service(settings: Settings = Depends(get_settings)) -> QuizService:
    """Dependency to get quiz service with ChromaDB repositories (stateless)."""
    from app.repositories.chroma_question_repository import ChromaQuestionRepository

    question_repo = ChromaQuestionRepository(settings)
    return QuizService(question_repo, settings)


@router.post("/quiz-me", response_model=QuizMeResponse)
async def quiz_me(
    req: QuizMeRequest, svc: QuizService = Depends(get_quiz_service)
) -> QuizMeResponse:
    """
    On receiving "Quiz Me", the system randomly selects one or more multiple-choice
    questions from MedMCQA already in memory and returns them with options.
    This endpoint does not track session state in MVP.
    """
    try:
        resp = await svc.quiz_me(req)
        return resp
    except Exception as e:
        logger.error(f"QuizMe failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/topics", response_model=dict[str, Any])
async def list_available_topics() -> dict[str, Any]:
    """
    List available topics/subjects/difficulties discovered during ingestion.

    This reads a lightweight catalog written by the ingestion job at
    `<CHROMA_PERSIST_DIR>/topics.json`. If the catalog does not exist yet,
    returns empty lists with a helpful message.
    """
    try:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        catalog_path = os.path.join(persist_dir, "topics.json")
        if not os.path.exists(catalog_path):
            return {
                "topics": [],
                "subjects": [],
                "difficulties": [],
                "counts": {"by_topic": {}, "by_subject": {}, "by_difficulty": {}},
                "version": "unknown",
                "note": "Catalog not found. Run the ingestion endpoint to populate topics.",
            }

        with open(catalog_path, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to read topics catalog: {e}")
        raise HTTPException(status_code=500, detail="Failed to load topics catalog")
