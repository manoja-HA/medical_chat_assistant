"""
Services package for Medical Quiz Assistant.
Contains business logic services for quiz, answer, and explanation functionality.
"""

# Important: Avoid importing heavy submodules at package import time.
# Importing modules like `ingest_service` can pull optional heavy dependencies
# (e.g., `datasets`, `langchain`) and slow down or break app startup.
#
# Consumers should import concrete services directly, e.g.:
#   from app.services.quiz_service import QuizService
# rather than importing from the package root.

__all__ = [
    # This list is informational only; we avoid importing the symbols eagerly.
    "QuizService",
    "AnswerService",
    "ExplainService",
    "RAGService",
    "IngestService",
    "EmbeddingService",
    "EmbeddingCache",
    "OllamaLLMService",
    "LLMCache",
    "CacheService",
    "CacheKeyBuilder",
    "AsyncProcessor",
    "BackgroundTaskManager",
    "PerformanceMonitor",
    "PerformanceOptimizer",
    "ConnectionPool",
]
