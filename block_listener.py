"""
Block listener module for Bitcoin mining system.

This module provides real-time block detection using ZMQ subscriptions
with fallback to polling for environments where ZMQ is not available.
"""

import time
import hashlib
from typing import Callable


# ZMQ configuration - will fallback if not available
ZMQ_AVAILABLE = False
try:
    import zmq

    ZMQ_AVAILABLE = True
except ImportError:
    print("Warning: ZMQ not available, using fallback polling method")


def listen_for_blocks(callback: Callable, verbose: bool = True) -> None:
    """
    Listen for new blocks using ZMQ or fallback polling

    Args:
        callback: Function to call when new block is detected
        verbose: Whether to output status messages
    """
    if ZMQ_AVAILABLE:
        _listen_zmq(callback, verbose)
    else:
        _listen_polling(callback, verbose)


def _listen_zmq(callback: Callable, verbose: bool) -> None:
    """
    Listen for blocks using ZMQ.

    Args:
        callback: Function to call when new block is detected
        verbose: Whether to output status messages
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    try:
        # Connect to Bitcoin Core ZMQ endpoints
        socket.connect("tcp://127.0.0.1:28332")  # hashblock port
        socket.setsockopt_string(zmq.SUBSCRIBE, "hashblock")
        socket.setsockopt_string(zmq.SUBSCRIBE, "rawblock")

        if verbose:
            print("ZMQ block listener started on tcp://127.0.0.1:28332")

        while True:
            try:
                topic, message = socket.recv_multipart(zmq.NOBLOCK)
                if verbose:
                    print(f"[ZMQ] Received {topic.decode()} notification")
                callback(topic, message)
            except zmq.Again:
                time.sleep(0.1)  # Brief sleep to prevent busy waiting
            except KeyboardInterrupt:
                if verbose:
                    print("ZMQ listener stopped by user")
                break

    except Exception as e:
        if verbose:
            print(f"ZMQ listener error: {e}")
    finally:
        socket.close()
        context.term()


def _listen_polling(callback: Callable, verbose: bool) -> None:
    """
    Fallback polling method when ZMQ is not available.

    Args:
        callback: Function to call when new block is detected
        verbose: Whether to output status messages
    """
    from mining_controller import get_blockchain_info

    if verbose:
        print("Using polling fallback for block detection")

    last_block_hash = None
    poll_interval = 10  # seconds

    while True:
        try:
            blockchain_info = get_blockchain_info()
            if blockchain_info.get("result"):
                current_hash = blockchain_info["result"].get("bestblockhash")

                if current_hash and current_hash != last_block_hash:
                    if last_block_hash is not None:  # Skip first iteration
                        if verbose:
                            print(f"[POLL] New block detected: {current_hash}")

                        # Create mock ZMQ-like message for callback compatibility
                        topic = b"hashblock"
                        message = current_hash.encode("utf-8")
                        callback(topic, message)

                    last_block_hash = current_hash

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            if verbose:
                print("Polling listener stopped by user")
            break
        except Exception as e:
            if verbose:
                print(f"Polling error: {e}")
            time.sleep(poll_interval)


def create_mock_block_data(block_hash: str) -> bytes:
    """
    Create mock block data from block hash for testing

    Args:
        block_hash: Block hash string

    Returns:
        Mock block data as bytes
    """
    # Create deterministic mock block data from hash
    hash_bytes = (
        bytes.fromhex(block_hash) if len(block_hash) == 64 else block_hash.encode()
    )
    mock_data = hashlib.sha256(hash_bytes + b"mock_block_data").digest()
    return mock_data * 8  # Make it larger for realistic processing
