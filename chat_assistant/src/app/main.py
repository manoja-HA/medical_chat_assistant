"""
Main application entry point for Medical Quiz Assistant.
Implements FastAPI application per ADR requirements.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from fastapi.responses import JSONResponse

from app.api import router as api_router
from app.core.config import get_settings, validate_settings
from app.core.middleware import setup_middleware
from app.core.observability import get_structured_logger, warm_up_ollama

# Get structured logger
logger = get_structured_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with ADR compliance."""
    settings = get_settings()

    # Startup
    logger.info("Starting Medical Quiz Assistant...")

    # Validate settings
    issues = validate_settings(settings)
    if issues:
        logger.warning(f"Configuration issues: {issues}")

    # Warm up Ollama if RAG is enabled (D11 compliance)
    app.state.rag_enabled = settings.rag_enabled
    if settings.rag_enabled:
        logger.info(f"Warming up Ollama model {settings.llm_model}...")
        ollama_ready = await warm_up_ollama(
            settings.ollama_host, settings.ollama_port, settings.llm_model
        )
        app.state.rag_enabled = ollama_ready
        if not ollama_ready:
            logger.warning("Ollama warm-up failed, RAG features disabled")
        else:
            logger.info("Ollama warm-up successful")
    else:
        logger.info("RAG disabled by configuration")

    # Store settings in app state
    app.state.settings = settings

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Medical Quiz Assistant...")
    logger.info("Shutdown complete")


# Get settings for app configuration
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
    A deterministic-first medical quiz assistant with optional RAG-enhanced explanations.
    """,
    version=settings.app_version,
    lifespan=lifespan,
)

# Get settings and setup middleware
settings = get_settings()
setup_middleware(app, settings)

# Include main API router
app.include_router(api_router)


@app.get("/", response_model=dict, tags=["Root"])
async def root():
    """
    Root endpoint with API information and navigation links.

    Returns comprehensive information about the API and available endpoints.
    """
    return {
        "name": "Medical Quiz Assistant",
        "version": "1.0.0",
        "description": "Deterministic-first medical quiz assistant with RAG compliance",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json",
        },
        "monitoring": {"health": "/healthz", "readiness": "/readyz", "metrics": "/metrics"},
        "api_endpoints": {
            "quiz": "/quiz/",
            "answer": "/answer/",
            "explain": "/explain/",
            "ingest": "/ingest/",
        },
        "features": [
            "Deterministic quiz questions",
            "Dataset-first explanations",
            "Optional RAG-enhanced explanations",
            "Session management",
            "Real-time monitoring",
        ],
    }


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404, content={"error": "Endpoint not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
