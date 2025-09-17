import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import RedirectResponse

from app.core.config import Settings, get_settings
from app.domain.schemas import IngestRequest, IngestResponse
from app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_ingest_service(settings: Settings = Depends(get_settings)) -> IngestService:
    """Dependency to get ingest service."""
    from app.repositories.question_repository import QuestionRepository
    from app.repositories.vector_repository import VectorRepository

    question_repo = QuestionRepository(settings)
    vector_repo = VectorRepository(settings)
    return IngestService(question_repo, vector_repo, settings)


@router.post("", response_model=IngestResponse, summary="Ingest Data")
async def ingest_data(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    ingest_service: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    """
    Ingest MedMCQA data into the system.

    Implements idempotent data loading with progress tracking.
    """
    try:
        logger.info(f"Starting data ingestion: split={request.split}, limit={request.limit}")

        start_time = time.time()

        # Process data ingestion
        result = await ingest_service.ingest_data(
            split=request.split, limit=request.limit, batch_size=request.batch_size
        )

        processing_time = int((time.time() - start_time) * 1000)

        response = IngestResponse(
            questions_processed=result["processed"],
            questions_skipped=result["skipped"],
            processing_time_ms=processing_time,
            dataset_version=result["version"],
        )

        logger.info(
            f"Data ingestion completed: {result['processed']} processed, "
            f"{result['skipped']} skipped in {processing_time}ms"
        )
        return response

    except ValueError as e:
        logger.warning(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in data ingestion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
