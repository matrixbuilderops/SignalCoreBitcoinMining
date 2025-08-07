"""
listener.py — Watches ZMQ, captures new block/hash

Core listener module for real-time Bitcoin block detection using ZMQ subscriptions.
Integrates with existing block_listener.py functionality.
"""

from block_listener import listen_for_blocks, create_mock_block_data
from typing import Callable, Optional
import time


class BitcoinBlockListener:
    """
    Main block listener class for ZMQ Bitcoin Core integration.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the Bitcoin block listener.

        Args:
            verbose: Enable verbose logging output
        """
        self.verbose = verbose
        self.callback_handler: Optional[Callable] = None
        self.running = False

    def set_callback(self, callback: Callable) -> None:
        """
        Set the callback function for block notifications.

        Args:
            callback: Function to call when new block is detected
        """
        self.callback_handler = callback

    def start_listening(self) -> None:
        """
        Start listening for new blocks via ZMQ.
        """
        if not self.callback_handler:
            raise ValueError("Callback handler must be set before starting listener")

        self.running = True
        if self.verbose:
            print("Starting Bitcoin block listener...")

        try:
            listen_for_blocks(self.callback_handler, self.verbose)
        except KeyboardInterrupt:
            if self.verbose:
                print("Block listener stopped by user")
        finally:
            self.running = False

    def stop_listening(self) -> None:
        """
        Stop the block listener.
        """
        self.running = False
        if self.verbose:
            print("Block listener stop requested")


def main():
    """
    Main entry point for testing the listener.
    """

    def test_callback(topic: bytes, message: bytes) -> None:
        """Test callback for block notifications."""
        block_hash = (
            message.decode("utf-8") if len(message) < 100 else message.hex()[:64]
        )
        print(f"Received block: {block_hash[:16]}...")

    # Create and start listener
    listener = BitcoinBlockListener(verbose=True)
    listener.set_callback(test_callback)

    try:
        listener.start_listening()
    except Exception as e:
        print(f"Listener error: {e}")


if __name__ == "__main__":
    main()
