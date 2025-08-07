"""
block_miner.py — Runs the actual block-solving loop

Main mining loop that coordinates all components: ZMQ listening, math processing,
AI analysis, and solution submission with continuous operation.
"""

import sys
import time
import signal
import threading
from typing import Dict, Any
from dataclasses import dataclass

from listener import BitcoinBlockListener
from model_caller import ModelCaller
from output_handler import OutputHandler, OutputMode
from submission_client import SubmissionClient
from math_module import process_block_with_math
from math_engine import run_level_16000


@dataclass
class MiningStats:
    """Mining statistics data structure."""

    blocks_detected: int = 0
    blocks_processed: int = 0
    solutions_generated: int = 0
    successful_submissions: int = 0
    start_time: float = 0
    last_block_time: float = 0


class BlockMiner:
    """
    Main block mining coordinator that runs the actual block-solving loop.
    """

    def __init__(self, verbose: bool = True, thinking_mode: bool = False):
        """
        Initialize the block miner.

        Args:
            verbose: Enable verbose output
            thinking_mode: Enable thinking mode for minimal output
        """
        # Initialize components
        self.output = OutputHandler(
            OutputMode.THINKING
            if thinking_mode
            else OutputMode.VERBOSE if verbose else OutputMode.QUIET
        )

        self.listener = BitcoinBlockListener(verbose=verbose)
        self.model_caller = ModelCaller(verbose=verbose)
        self.submission_client = SubmissionClient(verbose=verbose)

        # Mining state
        self.running = False
        self.stats = MiningStats()
        self._lock = threading.Lock()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.output.log_info("Shutdown signal received, stopping miner...", force=True)
        self.stop()

    def log(self, message: str, level: str = "info", force: bool = False) -> None:
        """
        Log message using output handler.

        Args:
            message: Message to log
            level: Log level
            force: Force output
        """
        if level == "error":
            self.output.log_error(message, force)
        elif level == "success":
            self.output.log_success(message, force)
        elif level == "warning":
            self.output.log_warning(message, force)
        else:
            self.output.log_info(message, force)

    def start(self) -> None:
        """
        Start the block mining operation.
        """
        self.log("Starting Bitcoin Block Miner...", force=True)
        self.stats.start_time = time.time()
        self.running = True

        # Verify system readiness
        if not self._verify_system_readiness():
            self.log("System readiness check failed", "error")
            return

        # Set up block callback
        self.listener.set_callback(self._on_block_detected)

        # Start mining loop
        self._run_mining_loop()

    def stop(self) -> None:
        """
        Stop the block mining operation.
        """
        with self._lock:
            self.running = False

        self.listener.stop_listening()
        self.output.stop_thinking()
        self.log("Block miner stopped", force=True)
        self._show_final_stats()

    def _verify_system_readiness(self) -> bool:
        """
        Verify all systems are ready for mining.

        Returns:
            True if all systems are ready
        """
        self.log("Verifying system readiness...")

        # Check submission client
        if not self.submission_client.verify_submission_readiness():
            self.log("Submission client not ready", "error")
            return False

        # Test model caller
        try:
            stats = self.model_caller.get_model_stats()
            self.log(f"Model caller ready - {stats['model_name']}")
        except Exception as e:
            self.log(f"Model caller error: {e}", "warning")
            # Continue anyway

        self.log("✓ All systems ready for mining", "success")
        return True

    def _run_mining_loop(self) -> None:
        """
        Run the main mining loop.
        """
        try:
            self.log("Starting block listener...")
            self.listener.start_listening()
        except KeyboardInterrupt:
            self.log("Mining loop interrupted by user", force=True)
        except Exception as e:
            self.log(f"Fatal error in mining loop: {e}", "error")
            sys.exit(1)

    def _on_block_detected(self, topic: bytes, message: bytes) -> None:
        """
        Handle new block detection and process mining opportunity.

        Args:
            topic: ZMQ topic
            message: Block data or hash
        """
        if not self.running:
            return

        with self._lock:
            self.stats.blocks_detected += 1
            self.stats.last_block_time = time.time()

        try:
            # Extract block information
            block_hash = (
                message.decode("utf-8") if len(message) < 100 else message.hex()[:64]
            )

            self.output.log_block_processing(block_hash, "detected")

            # Get block data
            if len(message) > 64:
                block_data = message
            else:
                from block_listener import create_mock_block_data

                block_data = create_mock_block_data(block_hash)

            # Process the mining opportunity
            self._process_mining_opportunity(block_data, block_hash)

        except Exception as e:
            self.log(f"Error processing block: {str(e)}", "error")

    def _process_mining_opportunity(self, block_data: bytes, block_hash: str) -> None:
        """
        Process complete mining opportunity through all stages.

        Args:
            block_data: Raw block data
            block_hash: Block hash
        """
        with self._lock:
            self.stats.blocks_processed += 1

        try:
            # Stage 1: Mathematical processing
            self.output.log_block_processing(block_hash, "math processing")
            processed_data, validation_results = process_block_with_math(
                block_data, 16000
            )

            # Stage 2: Level 16000 engine processing
            run_level_16000(verbose=False)

            # Stage 3: AI analysis
            self.output.log_block_processing(block_hash, "AI analysis")
            analysis = self.model_caller.analyze_mining_opportunity(
                validation_results, block_hash
            )

            # Stage 4: Decision and submission
            if analysis["should_proceed"]:
                self._attempt_submission(validation_results, analysis, block_hash)
            else:
                self.log(f"Mining opportunity declined - {analysis['recommendation']}")
                self.output.stop_thinking()

        except Exception as e:
            self.log(f"Error in mining opportunity processing: {str(e)}", "error")
            self.output.stop_thinking()

    def _attempt_submission(
        self,
        validation_results: Dict[str, Any],
        analysis: Dict[str, Any],
        block_hash: str,
    ) -> None:
        """
        Attempt to submit mining solution.

        Args:
            validation_results: Math validation results
            analysis: AI analysis results
            block_hash: Block hash
        """
        with self._lock:
            self.stats.solutions_generated += 1

        try:
            self.output.log_block_processing(block_hash, "submitting solution")

            # Submit solution
            submitted_hash = self.submission_client.submit_block_solution(
                validation_results
            )

            if submitted_hash:
                with self._lock:
                    self.stats.successful_submissions += 1

                self.output.log_mining_result(
                    True,
                    submitted_hash,
                    f"Level {analysis['level']}, Score: {analysis['validation_score']:.1f}%",
                )

                # Log progress every successful submission
                self._log_progress()

            else:
                self.output.log_mining_result(
                    False, block_hash, "Submission validation failed"
                )

        except Exception as e:
            self.log(f"Error in submission attempt: {str(e)}", "error")
            self.output.stop_thinking()

    def _log_progress(self) -> None:
        """Log current mining progress."""
        runtime = time.time() - self.stats.start_time

        if runtime > 0:
            blocks_per_hour = (self.stats.blocks_processed / runtime) * 3600
            success_rate = (
                (self.stats.successful_submissions / self.stats.blocks_processed * 100)
                if self.stats.blocks_processed > 0
                else 0
            )

            progress_stats = {
                "Runtime": f"{runtime/60:.1f} minutes",
                "Blocks Detected": self.stats.blocks_detected,
                "Blocks Processed": self.stats.blocks_processed,
                "Solutions Generated": self.stats.solutions_generated,
                "Successful Submissions": self.stats.successful_submissions,
                "Success Rate": f"{success_rate:.2f}%",
                "Processing Rate": f"{blocks_per_hour:.1f} blocks/hour",
            }

            self.output.log_stats(progress_stats)

    def _show_final_stats(self) -> None:
        """Show final mining statistics."""
        runtime = time.time() - self.stats.start_time

        # Get component stats
        model_stats = self.model_caller.get_model_stats()
        submission_stats = self.submission_client.get_submission_stats()

        print("\n" + "=" * 60)
        print("FINAL BITCOIN MINING STATISTICS")
        print("=" * 60)
        print(f"Total Runtime: {runtime/60:.1f} minutes")
        print(f"Blocks Detected: {self.stats.blocks_detected}")
        print(f"Blocks Processed: {self.stats.blocks_processed}")
        print(f"Solutions Generated: {self.stats.solutions_generated}")
        print(f"Successful Submissions: {self.stats.successful_submissions}")

        if self.stats.blocks_processed > 0:
            success_rate = (
                self.stats.successful_submissions / self.stats.blocks_processed * 100
            )
            print(f"Overall Success Rate: {success_rate:.2f}%")

        if runtime > 0:
            blocks_per_hour = (self.stats.blocks_processed / runtime) * 3600
            print(f"Processing Rate: {blocks_per_hour:.1f} blocks/hour")

        print(f"\nModel Calls: {model_stats['total_calls']}")
        print(f"Model Success Rate: {model_stats['success_rate_percent']}%")
        print(f"Submission Success Rate: {submission_stats['success_rate_percent']}%")
        print("=" * 60)

    def enable_thinking_mode(self) -> None:
        """Enable thinking mode for minimal output."""
        self.output.set_mode(OutputMode.THINKING)

    def enable_verbose_mode(self) -> None:
        """Enable verbose output mode."""
        self.output.set_mode(OutputMode.VERBOSE)

    def suppress_output(self) -> None:
        """Suppress all output except errors."""
        self.output.set_mode(OutputMode.ERROR_ONLY)

    def get_mining_stats(self) -> Dict[str, Any]:
        """
        Get current mining statistics.

        Returns:
            Dictionary with current statistics
        """
        runtime = time.time() - self.stats.start_time

        return {
            "runtime_minutes": runtime / 60,
            "blocks_detected": self.stats.blocks_detected,
            "blocks_processed": self.stats.blocks_processed,
            "solutions_generated": self.stats.solutions_generated,
            "successful_submissions": self.stats.successful_submissions,
            "success_rate_percent": (
                (self.stats.successful_submissions / self.stats.blocks_processed * 100)
                if self.stats.blocks_processed > 0
                else 0
            ),
            "blocks_per_hour": (
                (self.stats.blocks_processed / runtime * 3600) if runtime > 0 else 0
            ),
            "running": self.running,
        }


def main():
    """
    Main entry point for the block miner.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Bitcoin Block Mining Engine")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--thinking", action="store_true", help="Enable thinking... mode"
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    # Create miner
    miner = BlockMiner(verbose=not args.quiet, thinking_mode=args.thinking)

    if args.test:
        # Test mode - simulate some blocks
        print("Running in test mode...")

        # Simulate block detection
        test_blocks = [
            b"test_block_1_data_for_mining_engine_validation",
            b"test_block_2_data_for_mining_engine_validation",
            b"test_block_3_data_for_mining_engine_validation",
        ]

        for i, block_data in enumerate(test_blocks, 1):
            print(f"\nSimulating block {i}/3...")
            block_hash = f"00000000000{i:08d}abc123def456"
            miner._process_mining_opportunity(block_data, block_hash)
            time.sleep(1)

        stats = miner.get_mining_stats()
        print(f"\nTest Results: {stats}")

    else:
        # Normal operation
        try:
            miner.start()
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"Fatal error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
