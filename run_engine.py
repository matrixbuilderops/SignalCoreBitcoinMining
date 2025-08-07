"""
Central automation control loop for Bitcoin mining system.

This is the main engine that coordinates all components: listens via ZMQ,
triggers math + model cycles, submits solutions, and maintains continuous
operation with optional feedback modes.
"""

import sys
import time
import signal
import threading
from typing import Dict, Any, Optional
from block_listener import listen_for_blocks, create_mock_block_data
from model_orchestrator import ModelOrchestrator
from mining_controller import (
    submit_solution,
    monitor_mining_progress,
    get_blockchain_info,
)
from math_module import process_block_with_math


class BitcoinMiningEngine:
    """
    Central automation engine for continuous Bitcoin mining operations.
    """

    def __init__(self, verbose: bool = True, thinking_mode: bool = False):
        """
        Initialize the mining engine.

        Args:
            verbose: Enable verbose output
            thinking_mode: Enable minimal thinking... feedback mode
        """
        self.verbose = verbose
        self.thinking_mode = thinking_mode
        self.running = False
        self.total_blocks_processed = 0
        self.successful_submissions = 0
        self.engine_start_time = time.time()

        # Initialize model orchestrator
        self.model_orchestrator = ModelOrchestrator(
            verbose=verbose, thinking_mode=thinking_mode
        )

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def log(self, message: str, force: bool = False) -> None:
        """
        Log message with output control.

        Args:
            message: Message to log
            force: Force output even in quiet mode
        """
        if self.verbose or force:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] BitcoinEngine: {message}")
        elif self.thinking_mode:
            print("thinking...", end="", flush=True)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.log("Shutdown signal received, stopping engine...", force=True)
        self.stop()

    def start(self) -> None:
        """Start the continuous mining engine."""
        self.log("Starting Bitcoin Mining Engine...", force=True)
        self.running = True

        # Check initial system status
        self._check_system_status()

        # Start the main control loop
        self._run_control_loop()

    def stop(self) -> None:
        """Stop the mining engine."""
        self.running = False
        self.log("Mining engine stopped", force=True)
        self._show_final_stats()

    def _check_system_status(self) -> None:
        """Check system status before starting operations."""
        self.log("Checking system status...")

        try:
            # Check blockchain connection
            blockchain_info = get_blockchain_info()
            if blockchain_info.get("result"):
                height = blockchain_info["result"].get("blocks", 0)
                self.log(f"Blockchain connection OK - Height: {height}")
            else:
                self.log("Warning: Could not connect to blockchain", force=True)

            # Check mining progress
            progress = monitor_mining_progress()
            if progress.get("network_active"):
                self.log("Network status: Active and synchronized")
            else:
                self.log("Warning: Network may not be synchronized", force=True)

        except Exception as e:
            self.log(f"System check error: {e}", force=True)

    def _run_control_loop(self) -> None:
        """Main control loop that coordinates all operations."""
        self.log("Starting main control loop...")

        try:
            # Start block listener in the control loop
            listen_for_blocks(self._on_block_received, self.verbose)
        except KeyboardInterrupt:
            self.log("Control loop interrupted by user", force=True)
        except Exception as e:
            self.log(f"Fatal error in control loop: {e}", force=True)
            sys.exit(1)

    def _on_block_received(self, topic: bytes, message: bytes) -> None:
        """
        Handle new block detection and trigger math + model cycle.

        Args:
            topic: ZMQ topic (e.g., b'hashblock')
            message: Block hash or raw block data
        """
        if not self.running:
            return

        try:
            # Extract block information
            block_hash = (
                message.decode("utf-8") if len(message) < 100 else message.hex()[:64]
            )
            self.log(f"Processing new block: {block_hash[:16]}...")

            # Get block data
            if len(message) > 64:
                block_data = message
            else:
                block_data = create_mock_block_data(block_hash)

            self.total_blocks_processed += 1

            # Trigger math + model cycle
            self._process_mining_cycle(block_data, block_hash)

            # Log progress periodically
            if self.total_blocks_processed % 10 == 0:
                self._log_progress_update()

        except Exception as e:
            self.log(f"Error processing block: {str(e)}")

    def _process_mining_cycle(self, block_data: bytes, block_hash: str) -> None:
        """
        Process complete mining cycle: math validation + AI analysis + submission.

        Args:
            block_data: Raw block data
            block_hash: Block hash for context
        """
        try:
            # Step 1: Generate solution using model orchestrator
            self.log("Triggering math + model cycle...")
            solution = self.model_orchestrator.generate_solution(block_data, block_hash)

            if solution and solution.get("valid"):
                # Step 2: Submit solution if valid
                self._submit_mining_solution(solution)
            else:
                self.log("No valid solution generated for this block")

        except Exception as e:
            self.log(f"Error in mining cycle: {str(e)}")

    def _submit_mining_solution(self, solution: Dict[str, Any]) -> None:
        """
        Submit mining solution to the Bitcoin network.

        Args:
            solution: Valid solution from model orchestrator
        """
        try:
            self.log("Submitting mining solution...")

            # Convert solution to validation format for mining controller
            validation_results = {
                "level": solution["level"],
                "fork_integrity": True,  # Already validated in orchestrator
                "entropy_parity": True,
                "fork_sync": True,
                "sorrell": solution["sorrell"],
                "fork_cluster": solution["fork_cluster"],
                "over_recursion": solution["over_recursion"],
                "bit_load": solution["bit_load"],
                "cycles": solution["cycles"],
            }

            # Submit to network
            submitted_hash = submit_solution(validation_results)

            if submitted_hash:
                self.successful_submissions += 1
                self.log(f"Mining successful! Block: {submitted_hash}", force=True)
                self._log_successful_submission(solution, submitted_hash)
            else:
                self.log("Mining submission failed network validation")

        except Exception as e:
            self.log(f"Error submitting solution: {str(e)}")

    def _log_successful_submission(
        self, solution: Dict[str, Any], submitted_hash: str
    ) -> None:
        """Log details of successful submission."""
        if self.verbose:
            self.log(f"Successful submission details:")
            self.log(f"  Block Hash: {solution['block_hash']}")
            self.log(f"  Submitted Hash: {submitted_hash}")
            self.log(f"  Level: {solution['level']}")
            self.log(f"  AI Recommendation: {solution['recommendation']}")

    def _log_progress_update(self) -> None:
        """Log periodic progress updates."""
        if self.verbose:
            runtime = time.time() - self.engine_start_time
            success_rate = (
                (self.successful_submissions / self.total_blocks_processed * 100)
                if self.total_blocks_processed > 0
                else 0
            )
            blocks_per_hour = (
                (self.total_blocks_processed / runtime * 3600) if runtime > 0 else 0
            )

            self.log(f"Progress Update:")
            self.log(f"  Blocks Processed: {self.total_blocks_processed}")
            self.log(f"  Successful Submissions: {self.successful_submissions}")
            self.log(f"  Success Rate: {success_rate:.2f}%")
            self.log(f"  Processing Rate: {blocks_per_hour:.1f} blocks/hour")
            self.log(f"  Runtime: {runtime/60:.1f} minutes")

    def _show_final_stats(self) -> None:
        """Show final statistics when engine stops."""
        runtime = time.time() - self.engine_start_time

        # Get model orchestrator stats
        model_stats = self.model_orchestrator.get_performance_stats()

        print("\n" + "=" * 50)
        print("FINAL MINING ENGINE STATISTICS")
        print("=" * 50)
        print(f"Total Runtime: {runtime/60:.1f} minutes")
        print(f"Blocks Processed: {self.total_blocks_processed}")
        print(f"Successful Submissions: {self.successful_submissions}")

        if self.total_blocks_processed > 0:
            success_rate = (
                self.successful_submissions / self.total_blocks_processed * 100
            )
            print(f"Success Rate: {success_rate:.2f}%")

        print(f"Model Solutions Generated: {model_stats['solutions_generated']}")
        print(f"Model Valid Solutions: {model_stats['valid_solutions']}")
        print(f"Model Success Rate: {model_stats['success_rate_percent']}%")
        print("=" * 50)

    def enable_thinking_mode(self) -> None:
        """Enable minimal thinking... feedback mode."""
        self.thinking_mode = True
        self.verbose = False
        self.model_orchestrator.enable_thinking_mode()
        self.log("Thinking mode enabled")

    def suppress_output(self) -> None:
        """Suppress all output unless explicitly set."""
        self.thinking_mode = False
        self.verbose = False
        self.model_orchestrator.suppress_output()

    def enable_verbose_mode(self) -> None:
        """Enable full verbose output mode."""
        self.thinking_mode = False
        self.verbose = True
        self.model_orchestrator.enable_verbose_mode()
        self.log("Verbose mode enabled")


def main():
    """Main entry point for the Bitcoin mining engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Bitcoin Mining Automation Engine")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--thinking", action="store_true", help="Enable thinking... mode"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with mock data"
    )
    args = parser.parse_args()

    # Create and configure engine
    engine = BitcoinMiningEngine(verbose=not args.quiet, thinking_mode=args.thinking)

    if args.test:
        # Test mode - process a few mock blocks and exit
        print("Running in test mode...")
        test_blocks = [
            (
                b"test_block_1",
                "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
            ),
            (
                b"test_block_2",
                "00000000839a8e6886ab5951d76f411475428afc90947ee320161bbf18eb6048",
            ),
            (
                b"test_block_3",
                "000000006a625f06636b8bb6ac7b960a8d03705d1ace08b1a19da3fdcc99ddbd",
            ),
        ]

        for i, (block_data, block_hash) in enumerate(test_blocks, 1):
            print(f"\nProcessing test block {i}/3...")
            engine._process_mining_cycle(block_data, block_hash)
            time.sleep(1)  # Brief pause between tests

        engine._show_final_stats()
    else:
        # Normal operation - continuous mining
        try:
            engine.start()
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"Fatal error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
