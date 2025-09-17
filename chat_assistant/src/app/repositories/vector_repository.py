"""
Vector repository for managing embeddings and similarity search.
Implements dual collection strategy per ADR-001.
"""

import logging
from typing import Any, Optional

import numpy as np

from app.core.config import Settings

logger = logging.getLogger(__name__)


class VectorRepository:
    """Repository for managing vector embeddings and similarity search."""

    def __init__(self, settings: Settings):
        self.settings = settings
        # In-memory storage for MVP (would use ChromaDB in production)
        self._collections = {
            "questions": {},
            "explanations": {}
        }
        self._embeddings = {}

    async def add_document(
        self,
        collection_name: str,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None
    ) -> bool:
        """
        Add document to collection.

        Args:
            collection_name: Name of collection
            doc_id: Document identifier
            content: Document content
            metadata: Document metadata
            embedding: Optional embedding vector

        Returns:
            True if added successfully
        """
        logger.debug(f"Adding document {doc_id} to collection {collection_name}")

        if collection_name not in self._collections:
            self._collections[collection_name] = {}

        # Generate embedding if not provided
        if embedding is None:
            embedding = await self._generate_embedding(content)

        # Store document
        self._collections[collection_name][doc_id] = {
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }

        logger.debug(f"Document {doc_id} added to {collection_name}")
        return True

    async def similarity_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Perform similarity search.

        Args:
            query: Search query
            collection_name: Name of collection
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of similar documents
        """
        logger.debug(f"Searching {collection_name} with query: {query}")

        if collection_name not in self._collections:
            logger.warning(f"Collection {collection_name} not found")
            return []

        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # Get all documents in collection
        documents = list(self._collections[collection_name].values())

        # Apply filters if provided
        if filters:
            documents = self._filter_documents(documents, filters)

        if not documents:
            logger.debug("No documents found after filtering")
            return []

        # Calculate similarities
        similarities = []
        for doc in documents:
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((similarity, doc))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in similarities[:k]]

        logger.debug(f"Found {len(results)} similar documents")
        return results

    async def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Placeholder implementation - would use actual embedding model
        # In production, this would call the embedding service
        logger.debug(f"Generating embedding for text: {text[:50]}...")

        # Simple hash-based embedding for MVP
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to float vector
        embedding = [float(b) / 255.0 for b in hash_bytes[:8]]  # 8-dimensional vector

        logger.debug(f"Generated {len(embedding)}-dimensional embedding")
        return embedding

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)

    def _filter_documents(
        self,
        documents: list[dict[str, Any]],
        filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Filter documents by metadata.

        Args:
            documents: List of documents
            filters: Filter criteria

        Returns:
            Filtered documents
        """
        filtered = []

        for doc in documents:
            metadata = doc.get("metadata", {})
            matches = True

            for key, value in filters.items():
                if metadata.get(key) != value:
                    matches = False
                    break

            if matches:
                filtered.append(doc)

        return filtered

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """
        Get statistics for a collection.

        Args:
            collection_name: Name of collection

        Returns:
            Collection statistics
        """
        if collection_name not in self._collections:
            return {"error": f"Collection {collection_name} not found"}

        documents = list(self._collections[collection_name].values())

        if not documents:
            return {
                "collection_name": collection_name,
                "document_count": 0,
                "avg_content_length": 0,
                "metadata_keys": []
            }

        # Calculate statistics
        content_lengths = [len(doc["content"]) for doc in documents]
        avg_content_length = sum(content_lengths) / len(content_lengths)

        # Get unique metadata keys
        metadata_keys = set()
        for doc in documents:
            metadata_keys.update(doc.get("metadata", {}).keys())

        return {
            "collection_name": collection_name,
            "document_count": len(documents),
            "avg_content_length": round(avg_content_length, 2),
            "metadata_keys": sorted(list(metadata_keys))
        }

    async def delete_document(self, collection_name: str, doc_id: str) -> bool:
        """
        Delete document from collection.

        Args:
            collection_name: Name of collection
            doc_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        logger.info(f"Deleting document {doc_id} from collection {collection_name}")

        if collection_name not in self._collections:
            logger.warning(f"Collection {collection_name} not found")
            return False

        if doc_id not in self._collections[collection_name]:
            logger.warning(f"Document {doc_id} not found in collection {collection_name}")
            return False

        # Remove document
        del self._collections[collection_name][doc_id]

        logger.info(f"Document {doc_id} deleted from {collection_name}")
        return True

    async def clear_collection(self, collection_name: str) -> bool:
        """
        Clear all documents from collection.

        Args:
            collection_name: Name of collection

        Returns:
            True if cleared successfully
        """
        logger.info(f"Clearing collection {collection_name}")

        if collection_name not in self._collections:
            logger.warning(f"Collection {collection_name} not found")
            return False

        # Clear collection
        self._collections[collection_name].clear()

        logger.info(f"Collection {collection_name} cleared")
        return True

    async def get_document(self, collection_name: str, doc_id: str) -> Optional[dict[str, Any]]:
        """
        Get document by ID.

        Args:
            collection_name: Name of collection
            doc_id: Document identifier

        Returns:
            Document or None if not found
        """
        if collection_name not in self._collections:
            return None

        return self._collections[collection_name].get(doc_id)

    async def list_collections(self) -> list[str]:
        """Get list of collection names."""
        return list(self._collections.keys())

    async def get_collection_size(self, collection_name: str) -> int:
        """Get number of documents in collection."""
        if collection_name not in self._collections:
            return 0

        return len(self._collections[collection_name])
