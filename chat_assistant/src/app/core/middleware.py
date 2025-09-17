"""
Middleware for Medical Quiz Assistant.
Implements security, rate limiting, and observability per ADR-001.
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from .config import Settings
from .observability import get_observability_service

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware per ADR-001 security requirements."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.requests: dict[str, list] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests."""
        client_ip = request.client.host
        current_time = time.time()

        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time
                for req_time in self.requests[client_ip]
                if current_time - req_time < 60  # Keep last minute
            ]
        else:
            self.requests[client_ip] = []

        # Check rate limit
        if len(self.requests[client_ip]) >= self.settings.rate_limit_per_minute:
            logger.warning(f"Rate limit exceeded for IP {client_ip}")
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded. Please try again later."
            )

        # Record request
        self.requests[client_ip].append(current_time)

        # Process request
        response = await call_next(request)
        return response


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Correlation ID middleware for request tracing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add correlation ID to requests."""
        # Generate or extract correlation ID
        correlation_id = request.headers.get("x-correlation-id") or str(uuid.uuid4())

        # Add to request state
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["x-correlation-id"] = correlation_id

        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Structured JSON logging middleware with observability integration."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.observability = get_observability_service()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add structured logging and metrics to requests."""
        start_time = time.time()
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        # Extract session info from request (do this only once)
        session_id, question_id = await self._extract_request_info(request)

        # Create log context
        log_context = {
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": request.client.host,
            "session_id": session_id,
            "question_id": question_id,
            "user_agent": request.headers.get("user-agent", ""),
            "timestamp": start_time,
        }

        # Log request start
        logger.info(json.dumps({"event": "request_started", **log_context}))

        # Process request
        status_code = 500
        error = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error = str(e)
            logger.error(json.dumps({"event": "request_error", "error": error, **log_context}))
            raise
        finally:
            # Calculate latency
            latency_seconds = time.time() - start_time
            latency_ms = latency_seconds * 1000

            # Record metrics
            self.observability.record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration=latency_seconds,
            )

            # Log request completion
            logger.info(
                json.dumps(
                    {
                        "event": "request_completed",
                        "status_code": status_code,
                        "latency_ms": round(latency_ms, 2),
                        "latency_seconds": round(latency_seconds, 4),
                        "error": error,
                        **log_context,
                    }
                )
            )

        return response

    async def _extract_request_info(self, request: Request) -> tuple[str, str]:
        """Extract session ID and question ID from request headers and path."""
        # Try headers first for session ID
        session_id = request.headers.get("x-session-id", "unknown")
        question_id = "unknown"

        # Extract question_id from path parameters if present
        if hasattr(request, "path_params"):
            question_id = request.path_params.get("question_id", "unknown")

        # Don't read the request body in middleware to avoid interference
        # The body will be processed by the actual endpoint handlers

        return session_id, question_id


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware per ADR-001 security requirements."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        # Documentation endpoints that need relaxed CSP
        self.docs_endpoints = ["/docs", "/redoc", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply security measures to requests."""
        # Add security headers
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy - configurable based on settings
        if self.settings.csp_strict:
            if request.url.path in self.docs_endpoints and self.settings.docs_csp_relaxed:
                # More permissive CSP for Swagger UI and ReDoc
                csp = (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data: https:; "
                    "connect-src 'self'; "
                    "font-src 'self' data:; "
                    "object-src 'none'; "
                    "base-uri 'self'"
                )
            else:
                # Strict CSP for regular endpoints
                csp = (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data: https:; "
                    "connect-src 'self'"
                )
            response.headers["Content-Security-Policy"] = csp

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics collection middleware per ADR-001 observability requirements."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.request_count = 0
        self.error_count = 0
        self.latency_sum = 0.0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for requests."""
        start_time = time.time()

        try:
            response = await call_next(request)
            self.request_count += 1

            # Track errors
            if response.status_code >= 400:
                self.error_count += 1

            return response
        except Exception:
            self.error_count += 1
            raise
        finally:
            # Track latency
            latency = time.time() - start_time
            self.latency_sum += latency

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics."""
        avg_latency = self.latency_sum / self.request_count if self.request_count > 0 else 0

        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "avg_latency_seconds": avg_latency,
            "avg_latency_ms": avg_latency * 1000,
        }


def setup_cors_middleware(app, settings: Settings):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )


def setup_middleware(app, settings: Settings):
    """Setup all middleware for the application."""
    # Add middleware in reverse order (last added is first executed)

    # Security middleware
    app.add_middleware(SecurityMiddleware, settings=settings)

    # Metrics middleware (now integrated into structured logging)
    if getattr(settings, "enable_metrics", True):
        app.add_middleware(MetricsMiddleware, settings=settings)

    # Structured logging middleware (replaces old LoggingMiddleware)
    app.add_middleware(StructuredLoggingMiddleware, settings=settings)

    # Correlation ID middleware
    app.add_middleware(CorrelationMiddleware)

    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware, settings=settings)

    # CORS middleware
    setup_cors_middleware(app, settings)
