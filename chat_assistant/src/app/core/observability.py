"""
Observability module for Medical Quiz Assistant.
Implements health checks, readiness checks, and metrics endpoints.
"""

import logging
from datetime import datetime
from typing import Any

import httpx
from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.domain.schemas import HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"]
)
ACTIVE_SESSIONS = Gauge("active_sessions_total", "Number of active sessions")
QUESTION_PROCESSING_TIME = Histogram("question_processing_seconds", "Question processing time")
RAG_ENABLED = Gauge("rag_enabled", "Whether RAG is enabled (1) or disabled (0)")


class ObservabilityService:
    """Service for managing observability features."""

    def __init__(self):
        self.startup_time = datetime.utcnow()

    async def health_check(self, version: str) -> HealthResponse:
        """
        Basic health check - returns app status without external dependencies.
        Fast check for load balancers.
        """
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            services={"app": True},
            version=version,
            details={
                "uptime_seconds": int((datetime.utcnow() - self.startup_time).total_seconds())
            },
        )

    async def readiness_check(
        self,
        version: str,
        chroma_host: str,
        chroma_port: int,
        ollama_host: str,
        ollama_port: int,
        rag_enabled: bool,
    ) -> ReadinessResponse:
        """
        Comprehensive readiness check - verifies external dependencies.
        """
        dependencies = {}
        overall_ready = True

        # Check ChromaDB
        chroma_status = await self._check_chroma(chroma_host, chroma_port)
        dependencies["chroma"] = chroma_status
        if not chroma_status["healthy"]:
            overall_ready = False

        # Check Ollama (only if RAG enabled)
        if rag_enabled:
            ollama_status = await self._check_ollama(ollama_host, ollama_port)
            dependencies["ollama"] = ollama_status
            if not ollama_status["healthy"]:
                overall_ready = False

            # Update RAG enabled metric
            RAG_ENABLED.set(1 if ollama_status["healthy"] else 0)
        else:
            dependencies["ollama"] = {"healthy": True, "message": "RAG disabled", "model": None}
            RAG_ENABLED.set(0)

        return ReadinessResponse(
            ready=overall_ready,
            timestamp=datetime.utcnow(),
            dependencies=dependencies,
            version=version,
        )

    async def _check_chroma(self, host: str, port: int) -> dict[str, Any]:
        """Check ChromaDB health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{host}:{port}/api/v1/heartbeat")
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "healthy": True,
                        "version": data.get("nanosecond heartbeat", "unknown"),
                        "response_time_ms": int(response.elapsed.total_seconds() * 1000),
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {response.status_code}",
                        "response_time_ms": int(response.elapsed.total_seconds() * 1000),
                    }
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return {"healthy": False, "error": str(e), "response_time_ms": None}

    async def _check_ollama(self, host: str, port: int) -> dict[str, Any]:
        """Check Ollama health and model availability."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if service is up
                response = await client.get(f"http://{host}:{port}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    return {
                        "healthy": True,
                        "models": model_names,
                        "model_count": len(models),
                        "response_time_ms": int(response.elapsed.total_seconds() * 1000),
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {response.status_code}",
                        "response_time_ms": int(response.elapsed.total_seconds() * 1000),
                    }
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {"healthy": False, "error": str(e), "response_time_ms": None}

    async def get_metrics(self) -> Response:
        """Get Prometheus metrics."""
        try:
            metrics_data = generate_latest()
            return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return Response(
                content="# Error generating metrics\n",
                status_code=500,
                media_type=CONTENT_TYPE_LATEST,
            )

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def set_active_sessions(self, count: int):
        """Update active sessions metric."""
        ACTIVE_SESSIONS.set(count)

    def record_question_processing_time(self, duration: float):
        """Record question processing time."""
        QUESTION_PROCESSING_TIME.observe(duration)


# Global observability service instance
observability_service = ObservabilityService()


def get_observability_service() -> ObservabilityService:
    """Get observability service instance."""
    return observability_service


async def warm_up_ollama(host: str, port: int, model: str) -> bool:
    """
    Warm up Ollama by attempting to load the specified model.
    Returns True if successful, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Check if model is available
            response = await client.get(f"http://{host}:{port}/api/tags")
            if response.status_code != 200:
                logger.warning(f"Cannot reach Ollama at {host}:{port}")
                return False

            data = response.json()
            models = data.get("models", [])
            available_models = [m.get("name", "") for m in models]

            if model not in available_models:
                logger.warning(f"Model {model} not found. Available models: {available_models}")
                return False

            # Try a simple generation to warm up the model
            warm_up_payload = {
                "model": model,
                "prompt": "Hello",
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1},
            }

            response = await client.post(f"http://{host}:{port}/api/generate", json=warm_up_payload)

            if response.status_code == 200:
                logger.info(f"Successfully warmed up Ollama model {model}")
                return True
            else:
                logger.warning(f"Failed to warm up model {model}: HTTP {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"Ollama warm-up failed: {e}")
        return False


def get_structured_logger() -> logging.Logger:
    """Get structured logger for JSON logging."""
    logger = logging.getLogger("rag_quiz_assistant")

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
