"""
Real embedding service for Medical Quiz Assistant.
Implements BGE and Nomic embedding models per ADR-001.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using various models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.model_name = settings.embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_loaded = False

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._model_loaded:
            return

        logger.info(f"Initializing embedding model: {self.model_name}")

        try:
            if self.model_name in ["bge-small-en", "bge-base-en", "bge-large-en"]:
                await self._load_bge_model()
            elif self.model_name == "nomic-embed-text":
                await self._load_nomic_model()
            elif self.model_name == "all-MiniLM-L6-v2":
                await self._load_sentence_transformer_model()
            else:
                logger.warning(
                    f"Unknown model {self.model_name}, falling back to sentence-transformers"
                )
                await self._load_sentence_transformer_model()

            self._model_loaded = True
            logger.info(f"Embedding model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            # Fallback to simple model
            await self._load_fallback_model()

    async def _load_bge_model(self) -> None:
        """Load BGE (BAAI General Embedding) model."""
        try:
            from FlagEmbedding import FlagLLM, FlagReranker

            # Use sentence-transformers for BGE models
            self.model = SentenceTransformer(f"BAAI/{self.model_name}")
            self.model = self.model.to(self.device)

            logger.info(f"BGE model {self.model_name} loaded on {self.device}")

        except ImportError:
            logger.warning("FlagEmbedding not available, using sentence-transformers")
            self.model = SentenceTransformer(f"sentence-transformers/{self.model_name}")
            self.model = self.model.to(self.device)

    async def _load_nomic_model(self) -> None:
        """Load Nomic embedding model."""
        try:
            # Nomic models are available through sentence-transformers
            self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1")
            self.model = self.model.to(self.device)

            logger.info(f"Nomic model loaded on {self.device}")

        except Exception as e:
            logger.warning(f"Error loading Nomic model: {e}, using fallback")
            await self._load_fallback_model()

    async def _load_sentence_transformer_model(self) -> None:
        """Load sentence-transformers model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.model = self.model.to(self.device)

            logger.info(f"SentenceTransformer model {self.model_name} loaded on {self.device}")

        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {e}")
            await self._load_fallback_model()

    async def _load_fallback_model(self) -> None:
        """Load fallback embedding model."""
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model = self.model.to(self.device)
            self.model_name = "all-MiniLM-L6-v2"

            logger.info("Fallback embedding model loaded")

        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
            raise RuntimeError("Could not load any embedding model")

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self._model_loaded:
            await self.initialize()

        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)

            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.settings.embedding_dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._model_loaded:
            await self.initialize()

        if not texts:
            return []

        try:
            # Process in batches to avoid memory issues
            batch_size = self.settings.embedding_batch_size
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_tensor=False)

                # Convert to list format
                for embedding in batch_embeddings:
                    if isinstance(embedding, np.ndarray):
                        embeddings.append(embedding.tolist())
                    else:
                        embeddings.append(embedding)

            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.settings.embedding_dimension for _ in texts]

    async def embed_question(self, question: dict[str, Any]) -> list[float]:
        """
        Generate embedding for a question with metadata.

        Args:
            question: Question dictionary with stem, options, metadata

        Returns:
            Embedding vector
        """
        # Combine question text and options
        text_parts = [question.get("stem", "")]
        text_parts.extend(question.get("options", []))

        # Add metadata context
        if question.get("subject"):
            text_parts.append(f"Subject: {question['subject']}")
        if question.get("topic"):
            text_parts.append(f"Topic: {question['topic']}")

        combined_text = " ".join(text_parts)
        return await self.embed_text(combined_text)

    async def embed_explanation(self, explanation: dict[str, Any]) -> list[float]:
        """
        Generate embedding for an explanation.

        Args:
            explanation: Explanation dictionary with content and metadata

        Returns:
            Embedding vector
        """
        text_parts = [explanation.get("content", "")]

        # Add metadata context
        if explanation.get("question_id"):
            text_parts.append(f"Question: {explanation['question_id']}")
        if explanation.get("type"):
            text_parts.append(f"Type: {explanation['type']}")

        combined_text = " ".join(text_parts)
        return await self.embed_text(combined_text)

    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self._model_loaded:
            await self.initialize()

        # Test with a simple text to get dimension
        test_embedding = await self.embed_text("test")
        return len(test_embedding)

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._model_loaded,
            "dimension": await self.get_embedding_dimension(),
            "batch_size": self.settings.embedding_batch_size,
        }

    async def health_check(self) -> dict[str, Any]:
        """Check if the embedding service is healthy."""
        try:
            if not self._model_loaded:
                await self.initialize()

            # Test embedding generation
            test_embedding = await self.embed_text("health check")

            return {
                "status": "healthy",
                "model_loaded": self._model_loaded,
                "embedding_dimension": len(test_embedding),
                "device": self.device,
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "model_loaded": self._model_loaded}


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""

    def __init__(self, max_size: int = 10000):
        self.cache: dict[str, list[float]] = {}
        self.max_size = max_size
        self.access_count: dict[str, int] = {}

    def get(self, text: str) -> Optional[list[float]]:
        """Get cached embedding."""
        if text in self.cache:
            self.access_count[text] = self.access_count.get(text, 0) + 1
            return self.cache[text]
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Cache embedding."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_text = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_text]
            del self.access_count[lru_text]

        self.cache[text] = embedding
        self.access_count[text] = 1

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": sum(self.access_count.values()) / max(len(self.cache), 1),
        }
