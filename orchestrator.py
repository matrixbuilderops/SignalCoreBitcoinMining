import sys
import time
from typing import Dict, Any

from block_listener import listen_for_blocks, create_mock_block_data
from math_module import process_block_with_math
from ai_interface import call_ai_model, extract_recommendation
from mining_controller import submit_solution, monitor_mining_progress


class BitcoinMiningOrchestrator:
    """Main orchestrator for autonomous Bitcoin mining operations"""

    def __init__(self, verbose: bool = True, ai_enabled: bool = True):
        self.verbose = verbose
        self.ai_enabled = ai_enabled
        self.blocks_processed = 0
        self.successful_submissions = 0
        self.last_block_time = time.time()

    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def on_block_received(self, topic: bytes, message: bytes) -> None:
        """
        Handle new block detection and orchestrate mining process

        Args:
            topic: ZMQ topic (e.g., b'hashblock')
            message: Block hash or raw block data
        """
        try:
            block_hash = message.decode('utf-8') if len(message) < 100 else message.hex()[:64]
            self.log(f"Processing new block: {block_hash[:16]}...")

            # Create or extract block data
            if len(message) > 64:
                block_data = message
            else:
                block_data = create_mock_block_data(block_hash)

            # Process block with mathematical validation
            processed_data, validation_results = process_block_with_math(block_data)
            self.blocks_processed += 1

            self.log(f"Math processing complete - Level: {validation_results['level']}")

            # Get AI recommendation if enabled
            recommendation = "PROCEED"  # Default
            if self.ai_enabled:
                ai_response = call_ai_model(validation_results, block_hash)
                recommendation = extract_recommendation(ai_response)
                self.log(f"AI recommendation: {recommendation}")

                if self.verbose:
                    # Show AI response in verbose mode
                    self.log(f"AI Response: {ai_response[:100]}...")

            # Process based on recommendation
            if recommendation == "PROCEED":
                self._attempt_mining(validation_results, block_hash)
            elif recommendation == "RETRY":
                self.log("AI suggested retry - reprocessing with adjusted parameters")
                # Could implement retry logic with adjusted level here
                self._attempt_mining(validation_results, block_hash)
            elif recommendation == "HOLD":
                self.log("AI suggested hold - skipping this block")
            else:
                self.log(f"AI error or unknown recommendation: {recommendation}")

            self.last_block_time = time.time()
            self._log_status()

        except Exception as e:
            self.log(f"Error processing block: {str(e)}")

    def _attempt_mining(self, validation_results: Dict[str, Any], block_hash: str) -> None:
        """Attempt to submit mining solution"""
        try:
            submitted_hash = submit_solution(validation_results)
            if submitted_hash:
                self.successful_submissions += 1
                self.log(f"Mining successful! Block: {submitted_hash}")
            else:
                self.log("Mining submission failed validation")
        except Exception as e:
            self.log(f"Mining submission error: {str(e)}")

    def _log_status(self) -> None:
        """Log current mining status"""
        if self.verbose and self.blocks_processed % 5 == 0:
            success_rate = (self.successful_submissions / self.blocks_processed * 100) \
                if self.blocks_processed > 0 else 0
            self.log(f"Status: {self.blocks_processed} blocks processed, "
                     f"{self.successful_submissions} successful ({success_rate:.1f}%)")

    def start_monitoring(self) -> None:
        """Start the autonomous mining process"""
        self.log("Starting Bitcoin mining orchestrator...")

        # Check initial network status
        try:
            progress = monitor_mining_progress()
            if progress.get("network_active"):
                self.log("Network status: Active and synchronized")
            else:
                self.log("Warning: Network may not be fully synchronized")
        except Exception as e:
            self.log(f"Could not check network status: {e}")

        # Start block listener
        try:
            self.log("Starting block listener...")
            listen_for_blocks(self.on_block_received, self.verbose)
        except KeyboardInterrupt:
            self.log("Mining orchestrator stopped by user")
        except Exception as e:
            self.log(f"Fatal error in orchestrator: {e}")
            sys.exit(1)


def main():
    """Main entry point for the mining orchestrator"""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Bitcoin Mining Engine")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI recommendations")
    args = parser.parse_args()

    orchestrator = BitcoinMiningOrchestrator(
        verbose=not args.quiet,
        ai_enabled=not args.no_ai
    )

    orchestrator.start_monitoring()


if __name__ == "__main__":
    main()
