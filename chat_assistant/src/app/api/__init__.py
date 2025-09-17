"""
API router aggregation for Medical Quiz Assistant.
Combines all API endpoints into a single router per ADR requirements.
"""

from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.core.observability import get_observability_service

from .answer import router as answer_router
from .explain import router as explain_router
from .ingest import router as ingest_router
from .quiz import router as quiz_router

# Create main API router
router = APIRouter()

# Include sub-routers with appropriate prefixes and tags
router.include_router(quiz_router, prefix="/v1/quiz", tags=["Quiz"])

router.include_router(answer_router, prefix="/v1/answer", tags=["Answer"])

router.include_router(explain_router, prefix="/v1/explain", tags=["Explain"])

router.include_router(ingest_router, prefix="/v1/ingest", tags=["Ingest"])


# Observability endpoints (not in sub-routers to avoid conflicts)
@router.get(
    "/healthz",
    tags=["Monitoring"],
    summary="Health Check",
    description="Fast health check endpoint for load balancers and container orchestration",
    response_description="Basic health status of the application",
)
async def health_check():
    """
    Fast health check for load balancers.

    Returns basic application health status without checking external dependencies.
    This endpoint should respond quickly and is suitable for load balancer health checks.
    """
    settings = get_settings()
    observability = get_observability_service()
    return await observability.health_check(settings.app_version)


@router.get(
    "/readyz",
    tags=["Monitoring"],
    summary="Readiness Check",
    description="Comprehensive readiness check including external dependencies",
    response_description="Detailed readiness status with dependency information",
)
async def readiness_check():
    """
    Comprehensive readiness check.

    Checks application readiness including external dependencies like:
    - ChromaDB connection
    - Ollama/LLM availability (if RAG enabled)
    - Redis connectivity

    Use this endpoint to determine if the application is ready to serve traffic.
    """
    settings = get_settings()
    observability = get_observability_service()
    return await observability.readiness_check(
        version=settings.app_version,
        chroma_host=settings.chroma_host,
        chroma_port=settings.chroma_port,
        ollama_host=settings.ollama_host,
        ollama_port=settings.ollama_port,
        rag_enabled=settings.rag_enabled,
    )


@router.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus Metrics",
    description="Application metrics in Prometheus format",
    response_description="Metrics data for monitoring and alerting",
)
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns application metrics in Prometheus format including:
    - HTTP request metrics
    - Quiz session statistics
    - RAG performance metrics
    - System resource usage
    """
    observability = get_observability_service()
    return await observability.get_metrics()


__all__ = ["router"]
