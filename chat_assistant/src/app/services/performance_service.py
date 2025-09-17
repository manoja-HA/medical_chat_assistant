"""
Performance monitoring service for Medical Quiz Assistant.
Implements performance optimization and monitoring per ADR-001.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional

import psutil

from app.core.config import Settings

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Service for monitoring and optimizing performance."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.latency_histogram: dict[str, list[float]] = defaultdict(list)
        self.error_counts: dict[str, int] = defaultdict(int)
        self.request_counts: dict[str, int] = defaultdict(int)
        self.start_time = time.time()
        self.lock = threading.Lock()

    def record_latency(self, operation: str, latency_ms: float) -> None:
        """Record operation latency."""
        with self.lock:
            self.metrics[f"{operation}_latency"].append(latency_ms)
            self.latency_histogram[operation].append(latency_ms)

    def record_error(self, operation: str, error_type: str) -> None:
        """Record operation error."""
        with self.lock:
            self.error_counts[f"{operation}_{error_type}"] += 1

    def record_request(self, endpoint: str) -> None:
        """Record API request."""
        with self.lock:
            self.request_counts[endpoint] += 1

    def get_latency_stats(self, operation: str) -> dict[str, float]:
        """Get latency statistics for an operation."""
        with self.lock:
            latencies = list(self.metrics[f"{operation}_latency"])

            if not latencies:
                return {"count": 0}

            latencies.sort()
            count = len(latencies)

            return {
                "count": count,
                "min": min(latencies),
                "max": max(latencies),
                "mean": sum(latencies) / count,
                "p50": latencies[count // 2],
                "p95": latencies[int(count * 0.95)],
                "p99": latencies[int(count * 0.99)],
            }

    def get_error_rate(self, operation: str) -> float:
        """Get error rate for an operation."""
        with self.lock:
            total_requests = sum(
                count for key, count in self.request_counts.items() if operation in key
            )

            total_errors = sum(
                count for key, count in self.error_counts.items() if operation in key
            )

            return total_errors / max(total_requests, 1)

    def get_system_metrics(self) -> dict[str, Any]:
        """Get system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage("/")

            # Network I/O
            network = psutil.net_io_counters()

            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
                "uptime_seconds": time.time() - self.start_time,
            }

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        with self.lock:
            # Calculate overall metrics
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            overall_error_rate = total_errors / max(total_requests, 1)

            # Get latency stats for all operations
            latency_stats = {}
            for operation in set(
                key.split("_latency")[0] for key in self.metrics.keys() if "_latency" in key
            ):
                latency_stats[operation] = self.get_latency_stats(operation)

            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "overall_error_rate": overall_error_rate,
                "latency_stats": latency_stats,
                "system_metrics": self.get_system_metrics(),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self.lock:
            self.metrics.clear()
            self.latency_histogram.clear()
            self.error_counts.clear()
            self.request_counts.clear()
            self.start_time = time.time()

        logger.info("Performance metrics reset")


class PerformanceOptimizer:
    """Service for performance optimization."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.optimization_rules: list[Callable] = []
        self.performance_thresholds = {
            "max_latency_p95": 1000,  # 1 second
            "max_error_rate": 0.05,  # 5%
            "max_cpu_usage": 80,  # 80%
            "max_memory_usage": 85,  # 85%
        }

    def add_optimization_rule(self, rule: Callable) -> None:
        """Add a performance optimization rule."""
        self.optimization_rules.append(rule)

    async def check_performance_thresholds(self, monitor: PerformanceMonitor) -> list[str]:
        """Check if performance thresholds are exceeded."""
        violations = []

        # Check latency thresholds
        for operation in ["quiz", "answer", "explain", "ingest"]:
            stats = monitor.get_latency_stats(operation)
            if stats.get("p95", 0) > self.performance_thresholds["max_latency_p95"]:
                violations.append(
                    f"{operation} P95 latency exceeds threshold: {stats.get('p95', 0)}ms"
                )

        # Check error rates
        for operation in ["quiz", "answer", "explain", "ingest"]:
            error_rate = monitor.get_error_rate(operation)
            if error_rate > self.performance_thresholds["max_error_rate"]:
                violations.append(f"{operation} error rate exceeds threshold: {error_rate:.2%}")

        # Check system metrics
        system_metrics = monitor.get_system_metrics()
        if "error" not in system_metrics:
            if system_metrics.get("cpu_percent", 0) > self.performance_thresholds["max_cpu_usage"]:
                violations.append(
                    f"CPU usage exceeds threshold: {system_metrics.get('cpu_percent', 0)}%"
                )

            memory_percent = system_metrics.get("memory", {}).get("percent", 0)
            if memory_percent > self.performance_thresholds["max_memory_usage"]:
                violations.append(f"Memory usage exceeds threshold: {memory_percent}%")

        return violations

    async def apply_optimizations(self, monitor: PerformanceMonitor) -> list[str]:
        """Apply performance optimizations based on current metrics."""
        optimizations_applied = []

        # Check thresholds
        violations = await self.check_performance_thresholds(monitor)

        if violations:
            logger.warning(f"Performance threshold violations: {violations}")

            # Apply optimization rules
            for rule in self.optimization_rules:
                try:
                    result = await rule(monitor)
                    if result:
                        optimizations_applied.append(result)
                except Exception as e:
                    logger.error(f"Error applying optimization rule: {e}")

        return optimizations_applied

    def create_auto_scaling_rule(self) -> Callable:
        """Create auto-scaling optimization rule."""

        async def auto_scaling_rule(monitor: PerformanceMonitor) -> Optional[str]:
            system_metrics = monitor.get_system_metrics()

            if "error" in system_metrics:
                return None

            cpu_percent = system_metrics.get("cpu_percent", 0)
            memory_percent = system_metrics.get("memory", {}).get("percent", 0)

            # Scale up if high resource usage
            if cpu_percent > 70 or memory_percent > 75:
                return "High resource usage detected - consider scaling up"

            # Scale down if low resource usage
            if cpu_percent < 30 and memory_percent < 50:
                return "Low resource usage detected - consider scaling down"

            return None

        return auto_scaling_rule

    def create_caching_rule(self) -> Callable:
        """Create caching optimization rule."""

        async def caching_rule(monitor: PerformanceMonitor) -> Optional[str]:
            # Check if latency is high for specific operations
            for operation in ["quiz", "answer", "explain"]:
                stats = monitor.get_latency_stats(operation)
                if stats.get("p95", 0) > 500:  # 500ms threshold
                    return f"High latency detected for {operation} - consider increasing cache TTL"

            return None

        return caching_rule


class ConnectionPool:
    """Connection pool for database and external services."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pools: dict[str, Any] = {}
        self.pool_stats: dict[str, dict[str, Any]] = {}

    async def get_redis_pool(self) -> Any:
        """Get Redis connection pool."""
        if "redis" not in self.pools:
            import redis.asyncio as redis

            self.pools["redis"] = redis.ConnectionPool.from_url(
                self.settings.redis_url, max_connections=20, retry_on_timeout=True
            )

            self.pool_stats["redis"] = {
                "max_connections": 20,
                "created_at": datetime.utcnow().isoformat(),
            }

        return self.pools["redis"]

    async def get_http_pool(self) -> Any:
        """Get HTTP connection pool."""
        if "http" not in self.pools:
            import httpx

            limits = httpx.Limits(
                max_keepalive_connections=20, max_connections=100, keepalive_expiry=30
            )

            self.pools["http"] = httpx.AsyncClient(limits=limits, timeout=30.0)

            self.pool_stats["http"] = {
                "max_connections": 100,
                "max_keepalive": 20,
                "created_at": datetime.utcnow().isoformat(),
            }

        return self.pools["http"]

    async def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return {"pools": self.pool_stats, "total_pools": len(self.pools)}

    async def close_all_pools(self) -> None:
        """Close all connection pools."""
        for pool_name, pool in self.pools.items():
            try:
                if hasattr(pool, "close"):
                    await pool.close()
                elif hasattr(pool, "aclose"):
                    await pool.aclose()

                logger.info(f"Closed {pool_name} connection pool")
            except Exception as e:
                logger.error(f"Error closing {pool_name} pool: {e}")

        self.pools.clear()
        self.pool_stats.clear()
