"""
Question repository for managing MedMCQA questions and metadata.
Implements dual collection strategy per ADR-001.
"""

import hashlib
import json
import logging
from typing import Any, Optional

from app.core.config import Settings
from app.domain.models import Question

logger = logging.getLogger(__name__)


class QuestionRepository:
    """Repository for managing questions and metadata."""

    def __init__(self, settings: Settings):
        self.settings = settings
        # In-memory storage for MVP (would use ChromaDB in production)
        self._questions: dict[str, Question] = {}
        self._questions_by_filters: dict[str, list[str]] = {}

    # --- New lightweight sync helpers for Quiz Me flow ---
    def all_ids(self) -> list[str]:
        """Return all question IDs currently loaded in memory."""
        return list(self._questions.keys())

    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a question by ID (synchronous helper)."""
        return self._questions.get(question_id)

    async def get_by_id(self, question_id: str) -> Optional[Question]:
        """
        Get question by ID.

        Args:
            question_id: Question identifier

        Returns:
            Question or None if not found
        """
        question = self._questions.get(question_id)
        if question:
            logger.debug(f"Retrieved question {question_id}")
        else:
            logger.debug(f"Question {question_id} not found")
        return question

    async def get_questions_by_filters(
        self,
        filters: dict[str, Any]
    ) -> list[Question]:
        """
        Get questions matching filters.

        Args:
            filters: Dictionary of filter criteria

        Returns:
            List of matching questions
        """
        logger.debug(f"Getting questions with filters: {filters}")

        # If no filters, return all questions
        if not filters:
            return list(self._questions.values())

        # Build filter key for caching
        filter_key = self._build_filter_key(filters)

        # Check cache first
        if filter_key in self._questions_by_filters:
            question_ids = self._questions_by_filters[filter_key]
            questions = [self._questions[qid] for qid in question_ids if qid in self._questions]
            logger.debug(f"Retrieved {len(questions)} questions from cache")
            return questions

        # Filter questions
        matching_questions = []
        for question in self._questions.values():
            if self._matches_filters(question, filters):
                matching_questions.append(question)

        # Cache results
        question_ids = [q.id for q in matching_questions]
        self._questions_by_filters[filter_key] = question_ids

        logger.debug(f"Found {len(matching_questions)} matching questions")
        return matching_questions

    def _build_filter_key(self, filters: dict[str, Any]) -> str:
        """Build cache key for filters."""
        sorted_items = sorted(filters.items())
        filter_str = json.dumps(sorted_items, sort_keys=True)
        return hashlib.md5(filter_str.encode()).hexdigest()

    def _matches_filters(self, question: Question, filters: dict[str, Any]) -> bool:
        """Check if question matches filters."""
        for key, value in filters.items():
            if key == "subject" and question.subject != value:
                return False
            elif key == "topic" and question.topic != value:
                return False
            elif key == "difficulty" and question.difficulty != value:
                return False
            elif key == "split" and question.split != value:
                return False

        return True

    async def add_question(self, question: Question) -> Question:
        """
        Add a new question.

        Args:
            question: Question to add

        Returns:
            Added question
        """
        logger.info(f"Adding question {question.id}")

        # Check if question already exists
        if question.id in self._questions:
            logger.warning(f"Question {question.id} already exists, updating instead")
            return await self.update_question(question)

        # Add question
        self._questions[question.id] = question

        # Clear filter cache
        self._questions_by_filters.clear()

        logger.info(f"Question {question.id} added successfully")
        return question

    async def update_question(self, question: Question) -> Question:
        """
        Update existing question.

        Args:
            question: Updated question

        Returns:
            Updated question
        """
        logger.info(f"Updating question {question.id}")

        if question.id not in self._questions:
            logger.warning(f"Question {question.id} not found, creating new one")
            return await self.add_question(question)

        # Update question
        self._questions[question.id] = question

        # Clear filter cache
        self._questions_by_filters.clear()

        logger.info(f"Question {question.id} updated successfully")
        return question

    async def delete_question(self, question_id: str) -> bool:
        """
        Delete question.

        Args:
            question_id: Question identifier

        Returns:
            True if deleted, False if not found
        """
        logger.info(f"Deleting question {question_id}")

        if question_id not in self._questions:
            logger.warning(f"Question {question_id} not found")
            return False

        # Remove question
        del self._questions[question_id]

        # Clear filter cache
        self._questions_by_filters.clear()

        logger.info(f"Question {question_id} deleted successfully")
        return True

    async def get_questions_by_split(self, split: str) -> list[Question]:
        """
        Get questions by dataset split.

        Args:
            split: Dataset split (train, validation, test)

        Returns:
            List of questions in split
        """
        questions = [
            q for q in self._questions.values()
            if q.split == split
        ]

        logger.debug(f"Retrieved {len(questions)} questions from {split} split")
        return questions

    async def get_question_count(self) -> int:
        """Get total number of questions."""
        return len(self._questions)

    async def get_question_count_by_split(self) -> dict[str, int]:
        """Get question count by split."""
        counts = {"train": 0, "validation": 0, "test": 0}

        for question in self._questions.values():
            if question.split in counts:
                counts[question.split] += 1

        return counts

    async def get_question_count_by_difficulty(self) -> dict[str, int]:
        """Get question count by difficulty."""
        counts = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}

        for question in self._questions.values():
            difficulty = question.difficulty or "unknown"
            if difficulty in counts:
                counts[difficulty] += 1

        return counts

    async def get_question_count_by_topic(self) -> dict[str, int]:
        """Get question count by topic."""
        counts = {}

        for question in self._questions.values():
            topic = question.topic or "unknown"
            counts[topic] = counts.get(topic, 0) + 1

        return counts

    async def search_questions(
        self,
        query: str,
        limit: int = 10
    ) -> list[Question]:
        """
        Search questions by text content.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching questions
        """
        logger.debug(f"Searching questions with query: {query}")

        query_lower = query.lower()
        matching_questions = []

        for question in self._questions.values():
            # Search in stem and options
            searchable_text = f"{question.stem} {' '.join(question.options)}".lower()

            if query_lower in searchable_text:
                matching_questions.append(question)

                if len(matching_questions) >= limit:
                    break

        logger.debug(f"Found {len(matching_questions)} matching questions")
        return matching_questions

    async def get_random_questions(
        self,
        count: int = 1,
        filters: Optional[dict[str, Any]] = None
    ) -> list[Question]:
        """
        Get random questions.

        Args:
            count: Number of questions to return
            filters: Optional filters to apply

        Returns:
            List of random questions
        """
        import random

        # Get filtered questions
        if filters:
            questions = await self.get_questions_by_filters(filters)
        else:
            questions = list(self._questions.values())

        # Return random sample
        if len(questions) <= count:
            return questions

        return random.sample(questions, count)

    async def bulk_add_questions(self, questions: list[Question]) -> int:
        """
        Add multiple questions in bulk.

        Args:
            questions: List of questions to add

        Returns:
            Number of questions added
        """
        logger.info(f"Adding {len(questions)} questions in bulk")

        added_count = 0
        for question in questions:
            if question.id not in self._questions:
                self._questions[question.id] = question
                added_count += 1
            else:
                logger.warning(f"Question {question.id} already exists, skipping")

        # Clear filter cache
        self._questions_by_filters.clear()

        logger.info(f"Added {added_count} new questions")
        return added_count

    # Port interface methods
    async def get_random(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        exclude_ids: Optional[list[str]] = None
    ) -> Optional[Question]:
        """Get a random question with optional filters (implements port interface)."""
        filters = {}
        if topic:
            filters["topic"] = topic
        if difficulty:
            filters["difficulty"] = difficulty

        questions = await self.get_random_questions(1, filters)
        if not questions:
            return None

        # Filter out excluded IDs
        if exclude_ids:
            questions = [q for q in questions if q.id not in exclude_ids]

        return questions[0] if questions else None

    async def save(self, question: Question) -> None:
        """Save a question (implements port interface)."""
        await self.add_question(question)

    async def get_by_topic(self, topic: str, limit: int = 10) -> list[Question]:
        """Get questions by topic (implements port interface)."""
        filters = {"topic": topic}
        questions = await self.get_questions_by_filters(filters)
        return questions[:limit]
