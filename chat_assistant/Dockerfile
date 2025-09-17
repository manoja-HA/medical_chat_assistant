# syntax=docker/dockerfile:1.7

FROM python:3.12.7-slim-bookworm AS builder

# Environment variables for build efficiency and determinism
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Install build toolchain minimally for MVP speed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry 2.2.0 to match your local environment
RUN pip install --no-cache-dir poetry==2.2.0

# Copy Poetry files and source code
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

# Configure Poetry for Docker environment
RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10

RUN poetry install --only=main --no-root --compile

########## runtime: lean ##########
FROM python:3.12.7-slim-bookworm AS runtime

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

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code from builder
COPY --from=builder /app/src ./src

RUN useradd -m -u 1000 user

# Switch to user context BEFORE creating directories
USER user

# Set environment variables for HuggingFace cache locations
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/home/user/.cache/huggingface/datasets \
    HF_HUB_CACHE=/home/user/.cache/huggingface/hub \
    CHROMA_PERSIST_DIR=/home/user/chroma_db

# Set working directory to user's home
WORKDIR $HOME/app

# Copy application files with proper ownership
COPY --chown=user:user --from=builder /app/src ./src

RUN mkdir -p /home/user/.cache/huggingface/datasets/downloads \
    /home/user/.cache/huggingface/datasets/extracted \
    /home/user/.cache/huggingface/transformers \
    /home/user/.cache/huggingface/hub \
    /home/user/chroma_db \
    /home/user/logs \
    /home/user/data

# Expose port for the application
EXPOSE 8000

# Healthcheck for readiness 
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --connect-timeout 5 http://127.0.0.1:8000/healthz || exit 1

# Run Gunicorn with Uvicorn workers 
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000", "--workers", "2", \
    "--timeout", "60", "app.main:app"]