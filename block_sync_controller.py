"""
Block Sync Controller
---------------------
Synchronizes the block mining lifecycle, interfacing with:
- ZMQ Listener
- Bitcoin Orchestrator
- Mining Core
- Integrity Layer
- Model Interface Layer

Responsibilities:
- Initiate mining on new block detection
- Coordinate mathematical validation
- Relay results for integrity checks and submission
"""

import time
from zmq_listener import listen_for_blocks
from model_interface import query_model
from integrity_layer import validate_block_math
from mining_core import mine_block
from bitcoin_orchestrator import submit_block_to_bitcoin


def sync_blockchain():
    print("[Controller] Block Sync Controller Initialized")
    for block_data in listen_for_blocks():
        print("[Controller] New block received")

        prompt = f"Run LEVEL 16000 math on: {block_data}"
        model_result = query_model(prompt)
        print("[Controller] Model math result received")

        if not validate_block_math(model_result):
            print("[Controller] Integrity check failed — skipping block")
            continue

        mined_block = mine_block(block_data, model_result)
        print("[Controller] Block mining complete")

        submit_block_to_bitcoin(mined_block)
        print("[Controller] Block submitted to Bitcoin network")

        time.sleep(0.5)  # throttle if needed


if __name__ == "__main__":
    sync_blockchain()
