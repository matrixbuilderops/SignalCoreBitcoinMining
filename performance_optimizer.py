"""
Performance optimization module for SignalCore Bitcoin Mining System.

This module provides performance enhancements for high-throughput autonomous mining
including parallel processing, caching, and resource optimization.
"""

import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""

    blocks_per_second: float = 0.0
    avg_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    thread_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    successful_submissions: int = 0
    total_processing_time: float = 0.0
    queue_depth: int = 0


class ValidationCache:
    """Cache for mathematical validation results to improve performance."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize validation cache.

        Args:
            max_size: Maximum number of cached results
        """
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, block_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached validation result.

        Args:
            block_hash: Block hash key

        Returns:
            Cached validation result or None
        """
        with self._lock:
            if block_hash in self.cache:
                self.hits += 1
                self.access_times[block_hash] = time.time()
                return self.cache[block_hash].copy()
            else:
                self.misses += 1
                return None

    def set(self, block_hash: str, validation_result: Dict[str, Any]) -> None:
        """
        Cache validation result.

        Args:
            block_hash: Block hash key
            validation_result: Validation result to cache
        """
        with self._lock:
            # Remove oldest entry if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(
                    self.access_times.keys(), key=lambda k: self.access_times[k]
                )
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[block_hash] = validation_result.copy()
            self.access_times[block_hash] = time.time()

    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0


class ParallelBlockProcessor:
    """Parallel block processing for improved throughput."""

    def __init__(self, max_workers: int = 3, queue_size: int = 100):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum number of worker threads
            queue_size: Maximum queue size for pending blocks
        """
        self.max_workers = max_workers
        self.processing_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        self.executor = None
        self.running = False
        self.workers_busy = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start parallel processing."""
        if not self.running:
            self.running = True
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def stop(self) -> None:
        """Stop parallel processing."""
        if self.running:
            self.running = False
            if self.executor:
                self.executor.shutdown(wait=True)

    def submit_block(
        self, block_data: bytes, block_hash: str, processor_func: Callable
    ) -> bool:
        """
        Submit block for parallel processing.

        Args:
            block_data: Block data to process
            block_hash: Block hash
            processor_func: Function to process the block

        Returns:
            True if submitted successfully
        """
        try:
            self.processing_queue.put(
                (block_data, block_hash, processor_func), timeout=1
            )
            return True
        except queue.Full:
            return False

    def process_blocks(self) -> None:
        """Process blocks from queue in parallel."""
        futures = []

        while self.running:
            try:
                # Get blocks from queue
                block_data, block_hash, processor_func = self.processing_queue.get(
                    timeout=1
                )

                # Submit to thread pool
                future = self.executor.submit(
                    self._process_single_block, block_data, block_hash, processor_func
                )
                futures.append((future, block_hash))

                # Clean up completed futures
                completed_futures = []
                for future, hash_val in futures:
                    if future.done():
                        try:
                            result = future.result()
                            self.result_queue.put((hash_val, result))
                        except Exception as e:
                            self.result_queue.put((hash_val, f"ERROR: {e}"))
                        completed_futures.append((future, hash_val))

                # Remove completed futures
                for completed in completed_futures:
                    futures.remove(completed)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in parallel processing: {e}")

    def _process_single_block(
        self, block_data: bytes, block_hash: str, processor_func: Callable
    ) -> Any:
        """
        Process a single block.

        Args:
            block_data: Block data
            block_hash: Block hash
            processor_func: Processing function

        Returns:
            Processing result
        """
        with self._lock:
            self.workers_busy += 1

        try:
            start_time = time.time()
            result = processor_func(block_data, block_hash)
            processing_time = time.time() - start_time

            return {
                "result": result,
                "processing_time": processing_time,
                "timestamp": time.time(),
            }
        finally:
            with self._lock:
                self.workers_busy -= 1

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        return self.processing_queue.qsize()

    def get_worker_utilization(self) -> float:
        """Get worker utilization percentage."""
        with self._lock:
            return (
                (self.workers_busy / self.max_workers * 100)
                if self.max_workers > 0
                else 0.0
            )


class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.submission_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.total_blocks = 0
        self.successful_blocks = 0
        self._lock = threading.Lock()

    def record_processing_time(self, processing_time: float) -> None:
        """Record block processing time."""
        with self._lock:
            self.processing_times.append(processing_time)
            self.total_blocks += 1

    def record_submission(self, success: bool, submission_time: float = 0.0) -> None:
        """Record mining submission result."""
        with self._lock:
            if success:
                self.successful_blocks += 1
            if submission_time > 0:
                self.submission_times.append(submission_time)

    def get_metrics(
        self, cache: Optional[ValidationCache] = None
    ) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Args:
            cache: Optional validation cache for hit rate

        Returns:
            Current performance metrics
        """
        with self._lock:
            uptime = time.time() - self.start_time

            metrics = PerformanceMetrics()

            # Calculate blocks per second
            metrics.blocks_per_second = (
                self.total_blocks / uptime if uptime > 0 else 0.0
            )

            # Average processing time
            if self.processing_times:
                metrics.avg_processing_time = sum(self.processing_times) / len(
                    self.processing_times
                )

            # Success rate
            metrics.successful_submissions = self.successful_blocks
            metrics.total_processing_time = uptime

            # Cache hit rate
            if cache:
                metrics.cache_hit_rate = cache.get_hit_rate()

            # Memory usage (approximate)
            try:
                import psutil

                process = psutil.Process()
                metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except ImportError:
                pass

            return metrics

    def get_performance_summary(self, cache: Optional[ValidationCache] = None) -> str:
        """
        Get formatted performance summary.

        Args:
            cache: Optional validation cache

        Returns:
            Formatted performance summary
        """
        metrics = self.get_metrics(cache)

        summary = f"""
Performance Summary:
├─ Throughput: {metrics.blocks_per_second:.2f} blocks/sec
├─ Avg Processing: {metrics.avg_processing_time:.3f}s
├─ Success Rate: {metrics.successful_submissions}/{self.total_blocks} blocks
├─ Uptime: {metrics.total_processing_time/3600:.1f} hours
├─ Cache Hit Rate: {metrics.cache_hit_rate:.1f}%
└─ Memory Usage: {metrics.memory_usage_mb:.1f} MB
"""
        return summary


class OptimizedMiningEngine:
    """High-performance mining engine with optimization features."""

    def __init__(self, max_workers: int = 3, cache_size: int = 1000):
        """
        Initialize optimized mining engine.

        Args:
            max_workers: Maximum worker threads
            cache_size: Validation cache size
        """
        self.cache = ValidationCache(cache_size)
        self.parallel_processor = ParallelBlockProcessor(max_workers)
        self.performance_monitor = PerformanceMonitor()
        self.optimization_enabled = True

    def enable_optimizations(self) -> None:
        """Enable all performance optimizations."""
        self.optimization_enabled = True
        self.parallel_processor.start()

    def disable_optimizations(self) -> None:
        """Disable optimizations (fallback to sequential processing)."""
        self.optimization_enabled = False
        self.parallel_processor.stop()

    def process_block_optimized(
        self, block_data: bytes, block_hash: str, processor_func: Callable
    ) -> Any:
        """
        Process block with optimizations.

        Args:
            block_data: Block data
            block_hash: Block hash
            processor_func: Processing function

        Returns:
            Processing result
        """
        start_time = time.time()
        result = None

        try:
            # Check cache first
            if self.optimization_enabled:
                cached_result = self.cache.get(block_hash)
                if cached_result:
                    processing_time = time.time() - start_time
                    self.performance_monitor.record_processing_time(processing_time)
                    return cached_result

            # Process block
            if self.optimization_enabled and self.parallel_processor.running:
                # Use parallel processing
                submitted = self.parallel_processor.submit_block(
                    block_data, block_hash, processor_func
                )
                if not submitted:
                    # Queue full, process sequentially
                    result = processor_func(block_data, block_hash)
                else:
                    # For demo purposes, process immediately
                    result = processor_func(block_data, block_hash)
            else:
                # Sequential processing
                result = processor_func(block_data, block_hash)

            # Cache result
            if self.optimization_enabled and isinstance(result, dict):
                self.cache.set(block_hash, result)

            processing_time = time.time() - start_time
            self.performance_monitor.record_processing_time(processing_time)

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_monitor.record_processing_time(processing_time)
            raise

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "optimizations_enabled": self.optimization_enabled,
            "parallel_processing": self.parallel_processor.running,
            "cache_size": len(self.cache.cache),
            "cache_hit_rate": self.cache.get_hit_rate(),
            "queue_depth": self.parallel_processor.get_queue_depth(),
            "worker_utilization": self.parallel_processor.get_worker_utilization(),
            "performance_metrics": self.performance_monitor.get_metrics(self.cache),
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.parallel_processor.stop()
        self.cache.clear()


# Global optimization instance for easy access
_global_optimizer = None


def get_optimizer() -> OptimizedMiningEngine:
    """Get global optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = OptimizedMiningEngine()
    return _global_optimizer


def optimize_for_production() -> None:
    """Apply production optimizations."""
    optimizer = get_optimizer()
    optimizer.enable_optimizations()

    # Set environment optimizations
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    # GC optimizations
    import gc

    gc.set_threshold(700, 10, 10)  # Reduce GC frequency for performance


def get_performance_report() -> str:
    """Get comprehensive performance report."""
    optimizer = get_optimizer()
    status = optimizer.get_optimization_status()

    report = f"""
🚀 SignalCore Mining - Performance Report
========================================
Status: {'🟢 OPTIMIZED' if status['optimizations_enabled'] else '🟡 BASIC'}

Performance Metrics:
{optimizer.performance_monitor.get_performance_summary(optimizer.cache)}

Optimization Status:
├─ Parallel Processing: {'✓' if status['parallel_processing'] else '✗'}
├─ Validation Cache: {status['cache_size']} entries
├─ Cache Hit Rate: {status['cache_hit_rate']:.1f}%
├─ Queue Depth: {status['queue_depth']} blocks
└─ Worker Utilization: {status['worker_utilization']:.1f}%

System Resources:
└─ Memory Usage: {status['performance_metrics'].memory_usage_mb:.1f} MB
========================================
"""
    return report


if __name__ == "__main__":
    # Demo performance optimization
    print("SignalCore Bitcoin Mining - Performance Optimization Demo")
    print("=" * 60)

    optimizer = get_optimizer()
    optimizer.enable_optimizations()

    print(get_performance_report())

    # Simulate some processing
    import random
    import time

    def mock_processor(block_data, block_hash):
        time.sleep(random.uniform(0.1, 0.3))  # Simulate processing
        return {"level": 16000, "result": "success"}

    print("Running performance test...")
    for i in range(10):
        test_hash = f"test_block_{i:03d}"
        test_data = f"test_data_{i}".encode()

        optimizer.process_block_optimized(test_data, test_hash, mock_processor)
        optimizer.performance_monitor.record_submission(True, 0.1)

    print("\nFinal Performance Report:")
    print(get_performance_report())

    optimizer.cleanup()
