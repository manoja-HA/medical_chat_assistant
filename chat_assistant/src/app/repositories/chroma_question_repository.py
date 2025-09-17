"""
Enhanced ChromaDB implementation of QuestionRepository with proper explanation handling.
Integrates with existing ChromaDB data from ingestion service and explanations collection.
"""

import logging
import random
from typing import Any, Optional

import chromadb
from chromadb.api.client import Client

from app.core.config import Settings
from app.domain.models import Question
from app.domain.ports import QuestionRepository

logger = logging.getLogger(__name__)


class ChromaQuestionRepository(QuestionRepository):
    """Enhanced ChromaDB implementation with explanation support."""

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

        # Cache for better performance
        self._question_cache: dict[str, Question] = {}
        self._cache_populated = False

    def _ensure_client(self) -> Client:
        """Ensure ChromaDB client is initialized."""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(path=self.persist_dir)
                logger.info(f"ChromaDB client initialized with persist_dir: {self.persist_dir}")
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
                logger.info(f"Connected to ChromaDB collection: {self.questions_collection_name}")
            except Exception as e:
                logger.error(f"Collection {self.questions_collection_name} not found: {e}")
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
                # Explanations are optional - don't raise error
                return None
        return self._explanations_collection

    async def _populate_cache_if_needed(self) -> None:
        """Populate question cache from ChromaDB with explanations."""
        if self._cache_populated:
            return

        try:
            questions_collection = self._ensure_questions_collection()
            explanations_collection = self._ensure_explanations_collection()

            # Get all questions from ChromaDB
            result = questions_collection.get(include=["documents", "metadatas"])

            if not result or not result.get("ids"):
                logger.warning("No questions found in ChromaDB collection")
                self._cache_populated = True
                return

            ids = result["ids"]
            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])

            logger.info(f"Loading {len(ids)} questions from ChromaDB into cache")

            # Get explanations if available
            explanations_map = {}
            if explanations_collection:
                try:
                    exp_result = explanations_collection.get(include=["documents", "metadatas"])
                    if exp_result and exp_result.get("ids"):
                        for i, exp_id in enumerate(exp_result["ids"]):
                            exp_doc = (
                                exp_result["documents"][i]
                                if i < len(exp_result["documents"])
                                else ""
                            )
                            exp_meta = (
                                exp_result["metadatas"][i]
                                if i < len(exp_result["metadatas"])
                                else {}
                            )

                            # Map explanation to question ID
                            question_id = exp_meta.get("question_id")
                            if question_id:
                                explanations_map[question_id] = {
                                    "content": exp_doc,
                                    "metadata": exp_meta,
                                }
                    logger.info(f"Loaded {len(explanations_map)} explanations")
                except Exception as e:
                    logger.warning(f"Failed to load explanations: {e}")

            # Convert ChromaDB data to Question objects with explanations
            for i, question_id in enumerate(ids):
                try:
                    doc = documents[i] if i < len(documents) else ""
                    metadata = metadatas[i] if i < len(metadatas) else {}

                    # Get explanation for this question
                    explanation_data = explanations_map.get(question_id)

                    # Parse the document format from ingestion service
                    question_obj = self._parse_chroma_document(
                        question_id, doc, metadata, explanation_data
                    )
                    if question_obj:
                        self._question_cache[question_id] = question_obj

                except Exception as e:
                    logger.warning(f"Failed to parse question {question_id}: {e}")
                    continue

            logger.info(
                f"Loaded {len(self._question_cache)} questions with explanations into cache"
            )
            self._cache_populated = True

        except Exception as e:
            logger.error(f"Failed to populate question cache: {e}")
            self._cache_populated = True  # Prevent infinite retries

    def _parse_chroma_document(
        self,
        question_id: str,
        document: str,
        metadata: dict[str, Any],
        explanation_data: Optional[dict[str, Any]] = None,
    ) -> Optional[Question]:
        """
        Parse ChromaDB document into Question object with explanations.

        Expected format from ingestion service:
        Document: "<question_text>\nOptions: <option_a> | <option_b> | <option_c> | <option_d>"
        Metadata: {"correct_idx": 2, "subject": "Medicine", "topic": "Cardiology", ...}
        Explanation: Raw explanation text from explanations collection
        """
        try:
            # Parse document text
            if "\nOptions:" in document:
                parts = document.split("\nOptions:", 1)
                question_text = parts[0].strip()
                options_text = parts[1].strip() if len(parts) > 1 else ""

                # Parse options
                options = [opt.strip() for opt in options_text.split("|")] if options_text else []
            else:
                question_text = document.strip()
                options = []

            # Ensure we have at least 2 options, pad with empty strings
            while len(options) < 4:
                options.append("")

            # Get metadata
            correct_idx = int(metadata.get("correct_idx", 0))
            subject = metadata.get("subject", "Unknown")
            topic = metadata.get("topic", "General")
            difficulty = metadata.get("difficulty", "medium")
            split = metadata.get("split", "train")

            # Parse explanation data
            expl_correct = None
            expl_by_option = None

            if explanation_data:
                explanation_content = explanation_data.get("content", "")
                explanation_metadata = explanation_data.get("metadata", {})

                # Set correct explanation
                if explanation_content and explanation_content.strip():
                    expl_correct = explanation_content.strip()

                # For now, we'll use the same explanation for all incorrect options
                # In a more sophisticated version, you could parse per-option explanations
                if expl_correct:
                    expl_by_option = {}
                    for idx in range(4):
                        if idx != correct_idx:
                            expl_by_option[idx] = f"This option is incorrect. {expl_correct}"

            # Create Question object with explanations
            question = Question(
                id=question_id,
                stem=question_text,
                options=options[:4],  # Ensure exactly 4 options
                correct_idx=max(0, min(correct_idx, 3)),  # Ensure valid index
                expl_correct=expl_correct,  # Now properly populated!
                expl_by_option=expl_by_option,  # Now properly populated!
                subject=subject,
                topic=topic,
                difficulty=difficulty,
                split=split,
            )

            return question

        except Exception as e:
            logger.error(f"Error parsing document for question {question_id}: {e}")
            return None

    async def get_by_id(self, question_id: str) -> Optional[Question]:
        """Get question by ID from ChromaDB with explanations."""
        try:
            await self._populate_cache_if_needed()

            # Try cache first
            if question_id in self._question_cache:
                logger.debug(f"Retrieved question {question_id} from cache with explanations")
                return self._question_cache[question_id]

            # Try ChromaDB directly if not in cache
            questions_collection = self._ensure_questions_collection()
            explanations_collection = self._ensure_explanations_collection()

            # Get question
            result = questions_collection.get(ids=[question_id], include=["documents", "metadatas"])

            if result and result.get("ids") and len(result["ids"]) > 0:
                doc = result["documents"][0] if result.get("documents") else ""
                metadata = result["metadatas"][0] if result.get("metadatas") else {}

                # Get explanation for this specific question
                explanation_data = None
                if explanations_collection:
                    try:
                        # Look for explanation with matching question_id
                        exp_result = explanations_collection.get(
                            where={"question_id": {"$eq": question_id}},
                            include=["documents", "metadatas"],
                            limit=1,
                        )

                        if (
                            exp_result
                            and exp_result.get("documents")
                            and len(exp_result["documents"]) > 0
                        ):
                            explanation_data = {
                                "content": exp_result["documents"][0],
                                "metadata": (
                                    exp_result["metadatas"][0]
                                    if exp_result.get("metadatas")
                                    else {}
                                ),
                            }
                            logger.debug(f"Found explanation for question {question_id}")
                    except Exception as e:
                        logger.warning(f"Failed to get explanation for question {question_id}: {e}")

                question = self._parse_chroma_document(question_id, doc, metadata, explanation_data)
                if question:
                    # Add to cache
                    self._question_cache[question_id] = question
                    logger.debug(f"Retrieved and cached question {question_id} with explanations")
                    return question

            logger.debug(f"Question {question_id} not found")
            return None

        except Exception as e:
            logger.error(f"Error retrieving question {question_id}: {e}")
            return None

    async def get_random(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        exclude_ids: Optional[list[str]] = None,
    ) -> Optional[Question]:
        """Get a random question with optional filters and explanations."""
        try:
            await self._populate_cache_if_needed()

            if not self._question_cache:
                logger.warning("No questions available in cache")
                return None

            # Filter questions
            candidates = []
            exclude_set = set(exclude_ids) if exclude_ids else set()

            for question in self._question_cache.values():
                # Skip excluded questions
                if question.id in exclude_set:
                    continue

                # Apply filters
                if topic and question.topic != topic:
                    continue
                if difficulty and question.difficulty != difficulty:
                    continue

                candidates.append(question)

            if not candidates:
                logger.debug(
                    f"No questions found with filters: topic={topic}, difficulty={difficulty}"
                )
                return None

            # Return random question (now includes explanations)
            selected = random.choice(candidates)
            logger.debug(
                f"Selected random question {selected.id} with explanation: {bool(selected.expl_correct)}"
            )
            return selected

        except Exception as e:
            logger.error(f"Error getting random question: {e}")
            return None

    async def get_random_questions(
        self, count: int = 1, filters: Optional[dict[str, Any]] = None
    ) -> list[Question]:
        """Get multiple random questions with explanations."""
        try:
            await self._populate_cache_if_needed()

            if not self._question_cache:
                logger.warning("No questions available in cache")
                return []

            # Build candidate list
            candidates = []
            for question in self._question_cache.values():
                if filters:
                    # Apply filters
                    if filters.get("topic") and question.topic != filters["topic"]:
                        continue
                    if filters.get("difficulty") and question.difficulty != filters["difficulty"]:
                        continue
                    if filters.get("subject") and question.subject != filters["subject"]:
                        continue

                candidates.append(question)

            if not candidates:
                logger.debug(f"No questions found with filters: {filters}")
                return []

            # Return random sample (now includes explanations)
            sample_size = min(count, len(candidates))
            selected = random.sample(candidates, sample_size)

            explanations_count = sum(1 for q in selected if q.expl_correct)
            logger.debug(
                f"Selected {len(selected)} random questions, {explanations_count} with explanations"
            )
            return selected

        except Exception as e:
            logger.error(f"Error getting random questions: {e}")
            return []

    # Keep all the other methods from the original implementation unchanged
    async def save(self, question: Question) -> None:
        """Save a question (not implemented for ChromaDB read-only repository)."""
        logger.warning("Save operation not supported in ChromaDB repository")
        raise NotImplementedError("ChromaDB repository is read-only")

    async def get_by_topic(self, topic: str, limit: int = 10) -> list[Question]:
        """Get questions by topic with explanations."""
        try:
            await self._populate_cache_if_needed()

            matching_questions = [q for q in self._question_cache.values() if q.topic == topic][
                :limit
            ]

            explanations_count = sum(1 for q in matching_questions if q.expl_correct)
            logger.debug(
                f"Found {len(matching_questions)} questions for topic {topic}, {explanations_count} with explanations"
            )
            return matching_questions

        except Exception as e:
            logger.error(f"Error getting questions by topic {topic}: {e}")
            return []

    async def get_question_count(self) -> int:
        """Get total number of questions."""
        try:
            await self._populate_cache_if_needed()
            return len(self._question_cache)
        except Exception as e:
            logger.error(f"Error getting question count: {e}")
            return 0

    async def get_question_count_by_difficulty(self) -> dict[str, int]:
        """Get question count by difficulty."""
        try:
            await self._populate_cache_if_needed()

            counts = {}
            for question in self._question_cache.values():
                difficulty = question.difficulty or "Unknown"
                counts[difficulty] = counts.get(difficulty, 0) + 1

            return counts

        except Exception as e:
            logger.error(f"Error getting question counts by difficulty: {e}")
            return {}

    async def get_question_count_by_topic(self) -> dict[str, int]:
        """Get question count by topic."""
        try:
            await self._populate_cache_if_needed()

            counts = {}
            for question in self._question_cache.values():
                topic = question.topic or "Unknown"
                counts[topic] = counts.get(topic, 0) + 1

            return counts

        except Exception as e:
            logger.error(f"Error getting question counts by topic: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check if ChromaDB is accessible."""
        try:
            client = self._ensure_client()
            collection = client.get_collection(self.questions_collection_name)
            # Try to count documents
            result = collection.count()
            logger.debug(f"ChromaDB health check: {result} questions available")
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False
