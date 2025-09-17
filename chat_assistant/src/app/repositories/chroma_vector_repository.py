"""
Fixed ChromaDB implementation - Corrected query format for where clauses.
"""

import logging
from typing import Any, Optional

import chromadb
from chromadb.api.client import Client

from app.core.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorRepository:
    """Fixed ChromaDB implementation with proper query formatting."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.persist_dir = settings.chroma_persist_dir or "./chroma_db"
        self.questions_collection_name = settings.chroma_collection_questions or "questions"
        self.explanations_collection_name = (
            settings.chroma_collection_explanations or "explanations"
        )

        # ChromaDB client (lazy initialization)
        self._client: Optional[Client] = None
        self._questions_collection = None
        self._explanations_collection = None

    def _ensure_client(self) -> Client:
        """Ensure ChromaDB client is initialized."""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(path=self.persist_dir)
                logger.info(
                    f"ChromaDB vector client initialized with persist_dir: {self.persist_dir}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}")
                raise RuntimeError(f"Cannot connect to ChromaDB: {e}")
        return self._client

    def _ensure_questions_collection(self):
        """Ensure questions collection is available."""
        if self._questions_collection is None:
            client = self._ensure_client()
            try:
                self._questions_collection = client.get_collection(self.questions_collection_name)
                logger.debug(f"Connected to questions collection: {self.questions_collection_name}")
            except Exception as e:
                logger.error(
                    f"Questions collection {self.questions_collection_name} not found: {e}"
                )
                raise ValueError(
                    f"ChromaDB collection {self.questions_collection_name} does not exist. Run ingestion first."
                )
        return self._questions_collection

    def _ensure_explanations_collection(self):
        """Ensure explanations collection is available."""
        if self._explanations_collection is None:
            client = self._ensure_client()
            try:
                self._explanations_collection = client.get_collection(
                    self.explanations_collection_name
                )
                logger.debug(
                    f"Connected to explanations collection: {self.explanations_collection_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Explanations collection {self.explanations_collection_name} not found: {e}"
                )
                return None
        return self._explanations_collection

    def _build_where_clause(self, filters: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """
        Build proper ChromaDB where clause format.

        ChromaDB expects: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$eq": "value2"}}]}
        Not: {"field1": {"$eq": "value1"}, "field2": {"$eq": "value2"}}
        """
        if not filters:
            return None

        # Filter out None values
        valid_filters = {k: v for k, v in filters.items() if v is not None}

        if not valid_filters:
            return None

        # Build conditions list
        conditions = []
        for key, value in valid_filters.items():
            conditions.append({key: {"$eq": value}})

        # Single condition - return as is
        if len(conditions) == 1:
            return conditions[0]

        # Multiple conditions - use $and
        return {"$and": conditions}

    async def similarity_search(
        self, query: str, collection_name: str, k: int = 5, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Perform similarity search with fixed query format.
        """
        try:
            # Get appropriate collection
            if collection_name == "questions":
                collection = self._ensure_questions_collection()
            elif collection_name == "explanations":
                collection = self._ensure_explanations_collection()
                if collection is None:
                    logger.warning("Explanations collection not available")
                    return []
            else:
                raise ValueError(f"Unknown collection: {collection_name}")

            # Build proper where clause
            where_clause = self._build_where_clause(filters)

            logger.debug(
                f"Searching {collection_name} with query: '{query[:50]}...', where: {where_clause}"
            )

            # Perform search with corrected where clause
            results = collection.query(
                query_texts=[query],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                logger.debug(f"No results found for query in {collection_name}")
                return []

            # Format results
            documents = []
            docs = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []
            ids = results.get("ids", [[]])[0] or []

            for i, doc in enumerate(docs):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 1.0
                doc_id = ids[i] if i < len(ids) else f"unknown_{i}"

                # Convert distance to similarity score
                similarity = 1.0 - distance if distance <= 1.0 else 0.0

                documents.append(
                    {
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "similarity": similarity,
                        "distance": distance,
                    }
                )

            logger.debug(f"Found {len(documents)} similar documents in {collection_name}")
            return documents

        except Exception as e:
            logger.error(f"Error in similarity search for {collection_name}: {e}")
            return []

    # Keep all other methods unchanged from the previous implementation
    async def add_document(
        self,
        collection_name: str,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """Add document to collection (read-only for MVP)."""
        logger.warning(
            f"Add document operation not supported in ChromaDB repository (collection: {collection_name})"
        )
        return False

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Get statistics for a collection."""
        try:
            if collection_name == "questions":
                collection = self._ensure_questions_collection()
            elif collection_name == "explanations":
                collection = self._ensure_explanations_collection()
                if collection is None:
                    return {
                        "collection_name": collection_name,
                        "document_count": 0,
                        "error": "Collection not available",
                    }
            else:
                return {"error": f"Unknown collection: {collection_name}"}

            # Get document count
            count = collection.count()

            # Get sample for metadata analysis
            sample_result = collection.get(limit=100, include=["metadatas"])

            metadata_keys = set()
            if sample_result and sample_result.get("metadatas"):
                for metadata in sample_result["metadatas"]:
                    if isinstance(metadata, dict):
                        metadata_keys.update(metadata.keys())

            return {
                "collection_name": collection_name,
                "document_count": count,
                "metadata_keys": sorted(list(metadata_keys)),
                "sample_size": len(sample_result.get("metadatas", [])) if sample_result else 0,
            }

        except Exception as e:
            logger.error(f"Error getting stats for collection {collection_name}: {e}")
            return {"collection_name": collection_name, "error": str(e)}

    async def get_document(self, collection_name: str, doc_id: str) -> Optional[dict[str, Any]]:
        """Get document by ID from collection."""
        try:
            if collection_name == "questions":
                collection = self._ensure_questions_collection()
            elif collection_name == "explanations":
                collection = self._ensure_explanations_collection()
                if collection is None:
                    return None
            else:
                logger.error(f"Unknown collection: {collection_name}")
                return None

            result = collection.get(ids=[doc_id], include=["documents", "metadatas"])

            if result and result.get("ids") and len(result["ids"]) > 0:
                return {
                    "id": doc_id,
                    "content": result["documents"][0] if result.get("documents") else "",
                    "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                }

            return None

        except Exception as e:
            logger.error(f"Error getting document {doc_id} from {collection_name}: {e}")
            return None

    async def list_collections(self) -> list[str]:
        """Get list of available collections."""
        try:
            client = self._ensure_client()
            collections = client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    async def get_collection_size(self, collection_name: str) -> int:
        """Get number of documents in collection."""
        try:
            if collection_name == "questions":
                collection = self._ensure_questions_collection()
            elif collection_name == "explanations":
                collection = self._ensure_explanations_collection()
                if collection is None:
                    return 0
            else:
                return 0

            return collection.count()

        except Exception as e:
            logger.error(f"Error getting size of collection {collection_name}: {e}")
            return 0

    async def health_check(self) -> bool:
        """Check if ChromaDB vector store is accessible."""
        try:
            client = self._ensure_client()

            # Check if questions collection exists
            questions_collection = client.get_collection(self.questions_collection_name)
            questions_count = questions_collection.count()

            logger.debug(f"ChromaDB vector health check: {questions_count} questions available")
            return True

        except Exception as e:
            logger.error(f"ChromaDB vector health check failed: {e}")
            return False

    async def search_explanations(
        self, query: str, limit: int = 3, metadata_filter: Optional[dict[str, Any]] = None
    ) -> list[tuple[str, float]]:
        """Search explanations by semantic similarity."""
        try:
            documents = await self.similarity_search(
                query=query, collection_name="explanations", k=limit, filters=metadata_filter
            )

            # Convert to expected format
            results = []
            for doc in documents:
                results.append((doc["content"], doc["similarity"]))

            logger.debug(f"Found {len(results)} explanation matches for query")
            return results

        except Exception as e:
            logger.error(f"Error searching explanations: {e}")
            return []
