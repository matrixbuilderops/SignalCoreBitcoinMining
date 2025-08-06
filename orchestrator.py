from block_listener import listen_for_blocks
from math_module import process_block_with_math
from ai_interface import call_ai_model
from mining_controller import submit_solution

def on_block_received(topic, message):
    print(f"[{topic.decode()}] New block detected")
    processed = process_block_with_math(message)
    response = call_ai_model(processed)
    print(f"AI Response: {response}")
    txid = submit_solution()
    print(f"Mined Block TXID: {txid}")

if __name__ == "__main__":
    listen_for_blocks(on_block_received)

