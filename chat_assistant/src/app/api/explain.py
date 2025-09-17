import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.domain.schemas import ExplainRequest, ExplainResponse
from app.services.explain_service import ExplainService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_explain_service(settings: Settings = Depends(get_settings)) -> ExplainService:
    """Dependency to get explain service with properly integrated RAG service."""
    from app.repositories.chroma_question_repository import ChromaQuestionRepository
    from app.repositories.chroma_vector_repository import ChromaVectorRepository
    from app.services.cache_service import CacheService
    from app.services.llm_service import OllamaLLMService
    from app.services.rag_service import RAGService

    # Initialize repositories
    question_repo = ChromaQuestionRepository(settings)

    # Initialize RAG service with proper dependencies if enabled
    rag_service = None
    if settings.rag_enabled:
        try:
            # Create all required services for RAG
            vector_repo = ChromaVectorRepository(settings)
            llm_service = OllamaLLMService(settings)
            cache_service = CacheService(settings)

            # Create RAG service with proper dependency injection
            rag_service = RAGService(
                vector_repo=vector_repo,
                llm_service=llm_service,
                cache_service=cache_service,
                settings=settings,
            )

            logger.info("RAG service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            logger.info("Explain service will use dataset-only mode")
            rag_service = None

    return ExplainService(question_repo, rag_service, settings)


@router.post("/", response_model=ExplainResponse)
async def get_explanation(
    request: ExplainRequest, explain_service: ExplainService = Depends(get_explain_service)
) -> ExplainResponse:
    """
    Get explanation for a question.

    Dataset-first approach: defaults to dataset mode, with optional RAG enhancement.
    """
    try:
        logger.info(
            f"Getting explanation for question {request.question_id}, mode {request.mode}"
        )

        response = await explain_service.get_explanation(request)

        logger.info(
            f"Explanation generated: source={response.source}, "
            f"confidence={response.confidence}, citations={len(response.citations)}, "
            f"warnings={len(response.warnings)}"
        )
        return response

    except ValueError as e:
        logger.warning(f"Explanation request failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in explanation endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
