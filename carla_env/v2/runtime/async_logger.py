"""Asynchronous logging utilities for CARLA environments."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AsyncLogger:
    """Asynchronous logger that buffers log messages and writes them in batches."""

    def __init__(
        self,
        log_level: int = logging.INFO,
        buffer_size: int = 1000,
        flush_interval: float = 1.0,
        max_queue_size: int = 10000,
    ):
        """Initialize the AsyncLogger.

        Args:
            log_level: Minimum log level to capture
            buffer_size: Number of messages to buffer before auto-flush
            flush_interval: Time interval in seconds between auto-flushes
            max_queue_size: Maximum size of the message queue
        """
        self.log_level = log_level
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size

        self._message_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._is_running = False

        # Statistics
        self._stats = {
            "messages_logged": 0,
            "messages_dropped": 0,
            "flushes_performed": 0,
            "start_time": 0.0,
        }

    def start(self) -> None:
        """Start the async logging worker thread."""
        if self._is_running:
            logger.warning("AsyncLogger is already running")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._is_running = True
        self._stats["start_time"] = time.time()
        self._worker_thread.start()
        logger.info("AsyncLogger started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the async logging worker thread.

        Args:
            timeout: Maximum time to wait for worker thread to finish
        """
        if not self._is_running:
            logger.warning("AsyncLogger is not running")
            return

        self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        self._is_running = False
        logger.info("AsyncLogger stopped")

    def log(
        self,
        level: int,
        message: str,
        extra_data: Dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> bool:
        """Add a log message to the async queue.

        Args:
            level: Log level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message text
            extra_data: Additional data to include with the log entry
            timestamp: Custom timestamp (defaults to current time)

        Returns:
            True if message was queued successfully, False if queue is full
        """
        if level < self.log_level:
            return True  # Message filtered by level

        if not self._is_running:
            logger.warning("AsyncLogger is not running, dropping message")
            return False

        log_entry = {
            "timestamp": timestamp or time.time(),
            "level": level,
            "level_name": logging.getLevelName(level),
            "message": message,
            "extra_data": extra_data or {},
        }

        try:
            self._message_queue.put_nowait(log_entry)
            self._stats["messages_logged"] += 1
            return True
        except queue.Full:
            self._stats["messages_dropped"] += 1
            return False

    def debug(self, message: str, **kwargs: Any) -> bool:
        """Log a debug message."""
        return self.log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs: Any) -> bool:
        """Log an info message."""
        return self.log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> bool:
        """Log a warning message."""
        return self.log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: Any) -> bool:
        """Log an error message."""
        return self.log(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs: Any) -> bool:
        """Log a critical message."""
        return self.log(logging.CRITICAL, message, kwargs)

    def flush(self) -> int:
        """Manually flush queued messages to the logger.

        Returns:
            Number of messages flushed
        """
        messages_flushed = 0

        while not self._message_queue.empty():
            try:
                log_entry = self._message_queue.get_nowait()
                self._write_log_entry(log_entry)
                messages_flushed += 1
            except queue.Empty:
                break

        if messages_flushed > 0:
            self._stats["flushes_performed"] += 1

        return messages_flushed

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics.

        Returns:
            Dictionary containing logging statistics
        """
        runtime = time.time() - self._stats["start_time"] if self._stats["start_time"] > 0 else 0

        return {
            **self._stats.copy(),
            "queue_size": self._message_queue.qsize(),
            "is_running": self._is_running,
            "runtime_seconds": runtime,
            "messages_per_second": self._stats["messages_logged"] / runtime if runtime > 0 else 0,
        }

    def _worker_loop(self) -> None:
        """Main worker loop that processes log messages."""
        last_flush_time = time.time()
        buffer = []

        while not self._stop_event.is_set():
            try:
                # Try to get a message with a short timeout
                try:
                    log_entry = self._message_queue.get(timeout=0.1)
                    buffer.append(log_entry)
                except queue.Empty:
                    pass

                current_time = time.time()
                should_flush = (
                    len(buffer) >= self.buffer_size or (current_time - last_flush_time) >= self.flush_interval
                )

                if should_flush and buffer:
                    self._flush_buffer(buffer)
                    buffer.clear()
                    last_flush_time = current_time
                    self._stats["flushes_performed"] += 1

            except Exception as e:
                logger.error(f"AsyncLogger worker error: {e}")

        # Final flush on shutdown
        if buffer:
            self._flush_buffer(buffer)

        # Process any remaining messages in queue
        self.flush()

    def _flush_buffer(self, buffer: list) -> None:
        """Flush a buffer of log entries to the logger."""
        for log_entry in buffer:
            self._write_log_entry(log_entry)

    def _write_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Write a single log entry to the standard logger."""
        try:
            level = log_entry["level"]
            message = log_entry["message"]
            timestamp = log_entry["timestamp"]
            extra_data = log_entry.get("extra_data", {})

            # Format the message with timestamp and extra data
            formatted_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}] {message}"

            if extra_data:
                extra_str = ", ".join(f"{k}={v}" for k, v in extra_data.items())
                formatted_message += f" | {extra_str}"

            # Write to standard logger
            logger.log(level, formatted_message)

        except Exception as e:
            logger.error(f"Failed to write log entry: {e}")

    def __enter__(self) -> AsyncLogger:
        """Context manager entry: start the logger."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit: stop the logger."""
        self.stop()


# Global instance for convenience
_global_async_logger: AsyncLogger | None = None


def get_global_async_logger() -> AsyncLogger:
    """Get the global AsyncLogger instance, creating it if necessary."""
    global _global_async_logger
    if _global_async_logger is None:
        _global_async_logger = AsyncLogger()
    return _global_async_logger


def setup_global_async_logger(**kwargs: Any) -> AsyncLogger:
    """Setup and start the global AsyncLogger instance."""
    global _global_async_logger
    _global_async_logger = AsyncLogger(**kwargs)
    _global_async_logger.start()
    return _global_async_logger


def shutdown_global_async_logger() -> None:
    """Shutdown the global AsyncLogger instance."""
    global _global_async_logger
    if _global_async_logger is not None:
        _global_async_logger.stop()
        _global_async_logger = None
