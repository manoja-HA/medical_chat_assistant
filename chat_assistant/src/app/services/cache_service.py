"""
Caching service for Medical Quiz Assistant.
Implements Redis caching and performance optimization per ADR-001.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import Settings

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching with Redis and in-memory fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis_client: Optional[Redis] = None
        self.in_memory_cache: dict[str, Any] = {}
        self.cache_ttl = settings.cache_ttl_seconds
        self.max_memory_size = settings.cache_size

    async def initialize(self) -> None:
        """Initialize cache service."""
        if self.settings.use_redis:
            await self._initialize_redis()
        else:
            logger.info("Using in-memory cache (Redis disabled)")

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url, encoding="utf-8", decode_responses=True
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to in-memory cache")
            self.redis_client = None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        try:
            if self.redis_client:
                return await self._get_from_redis(key)
            else:
                return self._get_from_memory(key)

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            if self.redis_client:
                return await self._set_in_redis(key, value, ttl)
            else:
                return self._set_in_memory(key, value, ttl)

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if self.redis_client:
                result = await self.redis_client.delete(key)
                return result > 0
            else:
                if key in self.in_memory_cache:
                    del self.in_memory_cache[key]
                    return True
                return False

        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting from Redis: {e}")
            return None

    async def _set_in_redis(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Set value in Redis."""
        try:
            ttl = ttl or self.cache_ttl
            serialized_value = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Error setting in Redis: {e}")
            return False

    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache."""
        if key in self.in_memory_cache:
            item = self.in_memory_cache[key]
            # Check if expired
            if item["expires_at"] > datetime.utcnow():
                return item["value"]
            else:
                del self.in_memory_cache[key]
        return None

    def _set_in_memory(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Set value in in-memory cache."""
        try:
            ttl = ttl or self.cache_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            # Clean up expired items if cache is full
            if len(self.in_memory_cache) >= self.max_memory_size:
                self._cleanup_expired_memory()

            self.in_memory_cache[key] = {"value": value, "expires_at": expires_at}
            return True
        except Exception as e:
            logger.error(f"Error setting in memory cache: {e}")
            return False

    def _cleanup_expired_memory(self) -> None:
        """Clean up expired items from in-memory cache."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, item in self.in_memory_cache.items() if item["expires_at"] <= now
        ]

        for key in expired_keys:
            del self.in_memory_cache[key]

        # If still full, remove oldest items
        if len(self.in_memory_cache) >= self.max_memory_size:
            sorted_items = sorted(self.in_memory_cache.items(), key=lambda x: x[1]["expires_at"])

            # Remove oldest 10% of items
            remove_count = max(1, len(sorted_items) // 10)
            for key, _ in sorted_items[:remove_count]:
                del self.in_memory_cache[key]

    async def get_or_set(
        self, key: str, factory_func, ttl: Optional[int] = None, *args, **kwargs
    ) -> Any:
        """
        Get value from cache or set it using factory function.

        Args:
            key: Cache key
            factory_func: Function to call if value not in cache
            ttl: Time to live in seconds
            *args, **kwargs: Arguments for factory function

        Returns:
            Cached or newly generated value
        """
        # Try to get from cache first
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value

        # Generate new value
        try:
            if asyncio.iscoroutinefunction(factory_func):
                new_value = await factory_func(*args, **kwargs)
            else:
                new_value = factory_func(*args, **kwargs)

            # Cache the new value
            await self.set(key, new_value, ttl)
            return new_value

        except Exception as e:
            logger.error(f"Error in get_or_set factory function: {e}")
            raise

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    return await self.redis_client.delete(*keys)
                return 0
            else:
                # For in-memory cache, we can't easily pattern match
                # This is a limitation of the simple implementation
                logger.warning("Pattern invalidation not supported for in-memory cache")
                return 0

        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {e}")
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.redis_client:
                info = await self.redis_client.info()
                return {
                    "type": "redis",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "0B"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "hit_rate": self._calculate_hit_rate(info),
                }
            else:
                return {
                    "type": "memory",
                    "size": len(self.in_memory_cache),
                    "max_size": self.max_memory_size,
                    "ttl": self.cache_ttl,
                }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    def _calculate_hit_rate(self, info: dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0

    async def health_check(self) -> dict[str, Any]:
        """Check cache service health."""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return {"status": "healthy", "type": "redis"}
            else:
                return {"status": "healthy", "type": "memory"}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close(self) -> None:
        """Close cache connections."""
        if self.redis_client:
            await self.redis_client.close()


class CacheKeyBuilder:
    """Utility for building consistent cache keys."""

    @staticmethod
    def question_key(question_id: str) -> str:
        """Build cache key for question."""
        return f"question:{question_id}"

    @staticmethod
    def session_key(session_id: str) -> str:
        """Build cache key for session."""
        return f"session:{session_id}"

    @staticmethod
    def embedding_key(text: str) -> str:
        """Build cache key for embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{text_hash}"

    @staticmethod
    def explanation_key(question_id: str, mode: str) -> str:
        """Build cache key for explanation."""
        return f"explanation:{question_id}:{mode}"

    @staticmethod
    def retrieval_key(query: str, filters: dict[str, Any]) -> str:
        """Build cache key for retrieval results."""
        filters_str = json.dumps(filters, sort_keys=True)
        combined = f"{query}:{filters_str}"
        combined_hash = hashlib.md5(combined.encode()).hexdigest()
        return f"retrieval:{combined_hash}"
