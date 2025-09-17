#!/usr/bin/env python3
"""
Test script for the new ChromaDB/LangChain ingest service.
This script tests the ingest functionality without running the full application.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.services.ingest_service import IngestService
from app.core.config import Settings
from app.repositories.question_repository import QuestionRepository
from app.repositories.vector_repository import VectorRepository

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_ingest")


async def test_ingest():
    """Test the new ingest service with a small sample."""
    logger.info("Starting ingest service test...")

    # Create settings
    settings = Settings()

    # Create mock repositories (we won't use them in the new implementation)
    question_repo = QuestionRepository(settings)
    vector_repo = VectorRepository(settings)

    # Create ingest service
    ingest_service = IngestService(question_repo, vector_repo, settings)

    try:
        # Test with a small limit to avoid long processing times
        logger.info("Testing ingest with validation split, limit=5...")
        result = await ingest_service.ingest_data(split="validation", limit=5, batch_size=2)

        logger.info(f"Ingest result: {result}")

        # Test getting stats
        stats = await ingest_service.get_ingestion_stats()
        logger.info(f"Ingestion stats: {stats}")

        logger.info("✅ Ingest service test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Ingest service test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_ingest())
