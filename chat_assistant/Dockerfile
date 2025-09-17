# syntax=docker/dockerfile:1.7

########## builder: deps & wheels (Poetry respected) ##########
FROM python:3.12.7-slim-bookworm AS builder  # Updated to latest stable Python for MVP security patches

# Environment variables for build efficiency and determinism
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Install build toolchain minimally for MVP speed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN pip install --no-cache-dir poetry==1.8.3  # Updated Poetry version for compatibility

# Copy Poetry files and source code
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

# Export requirements and build wheels for app and dependencies
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

########## runtime: lean ##########
FROM python:3.12.7-slim-bookworm AS runtime  # Updated to match builder

# Environment variables for runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    GUNICORN_WORKERS=2 GUNICORN_TIMEOUT=60 GUNICORN_BIND=0.0.0.0:8000

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Install wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/* && \
    rm -rf /wheels

# Create non-root user and set permissions
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app && \
    mkdir -p /app/logs /app/data/processed /app/data/medmcqa && \
    chown -R app:app /app
USER app

# Expose port for the application
EXPOSE 8000

# Healthcheck for readiness (simplified for MVP)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/readyz || exit 1

# Run Gunicorn with Uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
    "--bind", "${GUNICORN_BIND}", "--workers", "${GUNICORN_WORKERS}", \
    "--timeout", "${GUNICORN_TIMEOUT}", "app.main:app"]