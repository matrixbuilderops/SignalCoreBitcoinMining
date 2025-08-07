"""
output_handler.py — Deals with output control and logging

Provides centralized output control with minimal terminal output toggle support.
Handles different output modes: verbose, quiet, thinking animation, and error-only.
"""

import sys
import time
import threading
from typing import Optional, Dict, Any
from enum import Enum


class OutputMode(Enum):
    """Output mode enumeration."""

    VERBOSE = "verbose"
    QUIET = "quiet"
    THINKING = "thinking"
    ERROR_ONLY = "error_only"


class OutputHandler:
    """
    Centralized output handler for Bitcoin mining system.
    """

    def __init__(self, mode: OutputMode = OutputMode.VERBOSE):
        """
        Initialize the output handler.

        Args:
            mode: Output mode to use
        """
        self.mode = mode
        self.thinking_active = False
        self.thinking_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def set_mode(self, mode: OutputMode) -> None:
        """
        Set the output mode.

        Args:
            mode: New output mode
        """
        with self._lock:
            self.stop_thinking()
            self.mode = mode

    def log(self, message: str, level: str = "INFO", force: bool = False) -> None:
        """
        Log a message based on current output mode.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            force: Force output regardless of mode
        """
        with self._lock:
            if force or self._should_output(level):
                timestamp = time.strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {level}: {message}"
                print(formatted_message)
                sys.stdout.flush()

    def log_info(self, message: str, force: bool = False) -> None:
        """Log info message."""
        self.log(message, "INFO", force)

    def log_warning(self, message: str, force: bool = False) -> None:
        """Log warning message."""
        self.log(message, "WARNING", force)

    def log_error(self, message: str, force: bool = True) -> None:
        """Log error message (force=True by default)."""
        self.log(message, "ERROR", force)

    def log_debug(self, message: str, force: bool = False) -> None:
        """Log debug message."""
        self.log(message, "DEBUG", force)

    def log_success(self, message: str, force: bool = False) -> None:
        """Log success message."""
        self.log(message, "SUCCESS", force)

    def _should_output(self, level: str) -> bool:
        """
        Determine if message should be output based on mode and level.

        Args:
            level: Log level

        Returns:
            True if message should be output
        """
        if self.mode == OutputMode.VERBOSE:
            return True
        elif self.mode == OutputMode.QUIET:
            return False
        elif self.mode == OutputMode.THINKING:
            return level in ["ERROR", "SUCCESS"]
        elif self.mode == OutputMode.ERROR_ONLY:
            return level in ["ERROR", "WARNING"]
        return False

    def start_thinking(self) -> None:
        """
        Start thinking animation for minimal output mode.
        """
        if self.mode != OutputMode.THINKING:
            return

        with self._lock:
            if self.thinking_active:
                return

            self.thinking_active = True
            self.thinking_thread = threading.Thread(
                target=self._thinking_animation, daemon=True
            )
            self.thinking_thread.start()

    def stop_thinking(self) -> None:
        """
        Stop thinking animation.
        """
        with self._lock:
            if self.thinking_active:
                self.thinking_active = False
                if self.thinking_thread and self.thinking_thread.is_alive():
                    self.thinking_thread.join(timeout=1)
                print()  # New line after thinking dots
                sys.stdout.flush()

    def _thinking_animation(self) -> None:
        """
        Display thinking animation dots.
        """
        dots = 0
        while self.thinking_active:
            if dots == 0:
                print("thinking", end="", flush=True)
            print(".", end="", flush=True)
            dots += 1
            if dots > 3:
                print("\rthinking", end="", flush=True)
                dots = 0
            time.sleep(0.5)

    def log_block_processing(self, block_hash: str, stage: str) -> None:
        """
        Log block processing with stage information.

        Args:
            block_hash: Block hash being processed
            stage: Processing stage
        """
        if self.mode == OutputMode.THINKING:
            self.start_thinking()
        else:
            self.log_info(f"Block {block_hash[:16]}... - {stage}")

    def log_mining_result(
        self, success: bool, block_hash: str, details: str = ""
    ) -> None:
        """
        Log mining result with appropriate formatting.

        Args:
            success: Whether mining was successful
            block_hash: Block hash
            details: Additional details
        """
        self.stop_thinking()

        if success:
            message = f"✓ Mining SUCCESS - Block: {block_hash}"
            if details:
                message += f" - {details}"
            self.log_success(message, force=True)
        else:
            message = f"✗ Mining FAILED - Block: {block_hash[:16]}..."
            if details:
                message += f" - {details}"
            self.log_error(message)

    def log_stats(self, stats: Dict[str, Any], force_output: bool = False) -> None:
        """
        Log statistics with proper formatting.

        Args:
            stats: Statistics dictionary
            force_output: Force output regardless of mode
        """
        if not force_output and self.mode in [OutputMode.QUIET, OutputMode.THINKING]:
            return

        self.log_info("=== MINING STATISTICS ===", force_output)
        for key, value in stats.items():
            self.log_info(f"  {key}: {value}", force_output)
        self.log_info("=" * 25, force_output)

    def log_system_status(self, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log system status information.

        Args:
            status: Status message
            details: Optional details dictionary
        """
        self.log_info(f"System Status: {status}")
        if details and self.mode == OutputMode.VERBOSE:
            for key, value in details.items():
                self.log_debug(f"  {key}: {value}")

    def create_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """
        Create a text progress bar.

        Args:
            current: Current progress value
            total: Total value
            width: Width of progress bar

        Returns:
            Progress bar string
        """
        if total == 0:
            return "[" + "=" * width + "]"

        progress = current / total
        filled = int(width * progress)
        bar = "=" * filled + "-" * (width - filled)
        percentage = progress * 100

        return f"[{bar}] {percentage:.1f}% ({current}/{total})"

    def prompt_user(self, question: str, default: str = "y") -> str:
        """
        Prompt user for input (only in verbose mode).

        Args:
            question: Question to ask
            default: Default response

        Returns:
            User response or default if not in verbose mode
        """
        if self.mode != OutputMode.VERBOSE:
            return default

        try:
            response = input(f"{question} [{default}]: ").strip()
            return response if response else default
        except KeyboardInterrupt:
            return default
        except Exception:
            return default


# Global output handler instance
_output_handler = OutputHandler()


def get_output_handler() -> OutputHandler:
    """Get the global output handler instance."""
    return _output_handler


def set_output_mode(mode: OutputMode) -> None:
    """Set the global output mode."""
    _output_handler.set_mode(mode)


def log_info(message: str, force: bool = False) -> None:
    """Global log info function."""
    _output_handler.log_info(message, force)


def log_error(message: str, force: bool = True) -> None:
    """Global log error function."""
    _output_handler.log_error(message, force)


def log_success(message: str, force: bool = False) -> None:
    """Global log success function."""
    _output_handler.log_success(message, force)


def main():
    """
    Test the output handler functionality.
    """
    handler = OutputHandler(OutputMode.VERBOSE)

    print("Testing output handler modes...")

    # Test verbose mode
    handler.set_mode(OutputMode.VERBOSE)
    handler.log_info("This is verbose mode")
    handler.log_warning("This is a warning")
    handler.log_error("This is an error")

    # Test quiet mode
    print("\nSwitching to quiet mode...")
    handler.set_mode(OutputMode.QUIET)
    handler.log_info("This should not appear")
    handler.log_error("Only errors should appear")

    # Test thinking mode
    print("\nSwitching to thinking mode...")
    handler.set_mode(OutputMode.THINKING)
    handler.log_info("Processing block...")
    handler.start_thinking()
    time.sleep(3)
    handler.stop_thinking()
    handler.log_success("Block processed successfully!")

    # Test mining result logging
    handler.set_mode(OutputMode.VERBOSE)
    handler.log_mining_result(
        True,
        "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
        "Level 16000 validation passed",
    )
    handler.log_mining_result(False, "abcdef123456", "Validation failed")

    # Test stats logging
    test_stats = {
        "Blocks Processed": 100,
        "Successful Submissions": 15,
        "Success Rate": "15.0%",
        "Runtime": "30.5 minutes",
    }
    handler.log_stats(test_stats)


if __name__ == "__main__":
    main()
