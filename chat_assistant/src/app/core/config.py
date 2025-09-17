"""
Configuration management for Medical Quiz Assistant.
Implements settings per ADR requirements with Pydantic BaseSettings.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""

    # Application
    app_name: str = Field("Medical Quiz Assistant", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Logging level")
    enable_docs: bool = Field(True, description="Enable Swagger/OpenAPI documentation")

    # API
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, description="API port")
    api_workers: int = Field(1, description="Number of API workers")
    cors_origins: list[str] = Field(["*"], description="CORS allowed origins")

    # Rate Limiting
    rate_limit_per_minute: int = Field(60, description="Rate limit per minute per IP")
    rate_limit_burst: int = Field(10, description="Rate limit burst allowance")

    # Session Management (Redis)
    redis_host: str = Field("redis", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_db: int = Field(0, description="Redis database number")
    redis_ttl: int = Field(86400, description="Redis key TTL in seconds (24 hours)")
    session_timeout_hours: int = Field(24, description="Session timeout in hours")
    max_sessions: int = Field(1000, description="Maximum number of active sessions")

    # Vector Database (ChromaDB)
    chroma_host: str = Field("chroma", description="ChromaDB host")
    chroma_port: int = Field(8000, description="ChromaDB port")
    chroma_collection_questions: str = Field("questions", description="Questions collection name")
    chroma_collection_explanations: str = Field(
        "explanations", description="Explanations collection name"
    )
    chroma_persist_dir: str = Field(
        "./chroma_db", alias="CHROMA_PERSIST_DIR", description="ChromaDB persistence directory"
    )

    # Embeddings
    embedding_model: str = Field("bge-small-en", description="Embedding model name")
    embedding_dimension: int = Field(384, description="Embedding dimension")
    embedding_batch_size: int = Field(32, description="Embedding batch size")

    # RAG Configuration
    rag_enabled: bool = Field(True, description="Enable RAG explanations")
    rag_top_k: int = Field(5, description="Number of documents to retrieve for RAG")
    rag_confidence_threshold: float = Field(
        0.7, description="Minimum confidence for RAG explanations"
    )
    rag_max_tokens: int = Field(1000, description="Maximum tokens for RAG responses")

    # Reranker Configuration
    reranker_enabled: bool = Field(False, description="Enable reranker for retrieved documents")

    # LLM Configuration (Ollama) - D11 compliance
    ollama_host: str = Field("localhost", description="Ollama host")
    ollama_port: int = Field(11434, description="Ollama port")
    llm_model: str = Field("llama3.2:1b", description="LLM model name")
    llm_temperature: float = Field(0.1, description="LLM temperature")
    llm_max_tokens: int = Field(512, description="LLM max tokens")

    # Data Processing
    data_batch_size: int = Field(100, description="Data processing batch size")
    data_max_questions: Optional[int] = Field(None, description="Maximum questions to process")
    data_validation_split: float = Field(0.1, description="Validation split ratio")

    # Monitoring & Observability
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(False, description="Enable OpenTelemetry tracing")
    metrics_port: int = Field(9090, description="Metrics server port")

    # Security
    secret_key: str = Field("your-secret-key-here", description="Secret key for security")
    jwt_secret: str = Field("your-jwt-secret-here", description="JWT secret key")
    jwt_expire_hours: int = Field(24, description="JWT expiration in hours")
    csp_strict: bool = Field(False, description="Enable strict Content Security Policy")
    docs_csp_relaxed: bool = Field(False, description="Use relaxed CSP for documentation endpoints")

    # Performance
    cache_size: int = Field(1000, description="In-memory cache size")
    cache_ttl_seconds: int = Field(3600, description="Cache TTL in seconds")
    max_concurrent_requests: int = Field(100, description="Maximum concurrent requests")

    # Evaluation
    eval_golden_set_size: int = Field(100, description="Size of golden evaluation set")
    eval_recall_threshold: float = Field(0.8, description="Minimum recall threshold for CI")
    eval_mrr_threshold: float = Field(0.7, description="Minimum MRR threshold for CI")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


def validate_settings(settings: Settings) -> list[str]:
    """
    Validate settings and return list of issues.

    Returns:
        List of validation issues (empty if all valid)
    """
    issues = []

    # Validate numeric ranges
    if settings.rate_limit_per_minute <= 0:
        issues.append("Rate limit must be positive")

    if settings.rag_confidence_threshold < 0 or settings.rag_confidence_threshold > 1:
        issues.append("RAG confidence threshold must be between 0 and 1")

    if settings.llm_temperature < 0 or settings.llm_temperature > 2:
        issues.append("LLM temperature must be between 0 and 2")

    if settings.ollama_port <= 0 or settings.ollama_port > 65535:
        issues.append("Ollama port must be between 1 and 65535")

    if settings.chroma_port <= 0 or settings.chroma_port > 65535:
        issues.append("Chroma port must be between 1 and 65535")

    if settings.redis_port <= 0 or settings.redis_port > 65535:
        issues.append("Redis port must be between 1 and 65535")

    # Validate model names
    valid_embedding_models = ["bge-small-en", "nomic-embed-text", "all-MiniLM-L6-v2"]
    if settings.embedding_model not in valid_embedding_models:
        issues.append(
            f"Invalid embedding model: {settings.embedding_model}. Valid options: {valid_embedding_models}"
        )

    # Validate LLM model format (ollama models can be custom)
    if not settings.llm_model or len(settings.llm_model.strip()) == 0:
        issues.append("LLM model name cannot be empty")

    return issues
