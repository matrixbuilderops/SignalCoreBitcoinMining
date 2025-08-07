# Core mining orchestration script - entrypoint

from zmq_listener import start_zmq_listener
from model_interface import interact_with_model
from mining_engine import start_mining_process
from rpc_client import get_block_template
from utils import log_event, SYSTEM_CONFIG


def main():
    log_event("[INIT] SignalCore Bitcoin Mining System Starting")

    # Step 1: Start ZMQ listener for all configured streams
    zmq_threads = start_zmq_listener(SYSTEM_CONFIG['zmq_endpoints'])

    # Step 2: Retrieve block template via RPC
    block_template = get_block_template()
    if not block_template:
        log_event("[ERROR] Failed to retrieve block template.")
        return

    # Step 3: Initiate model interaction for recursion mining logic
    model_input = {
        "block_template": block_template,
        "level": SYSTEM_CONFIG['recursion_level'],
        "math_model": SYSTEM_CONFIG['math_equation'],
    }
    solution = interact_with_model(model_input)
    if not solution:
        log_event("[ERROR] Model interaction failed.")
        return

    # Step 4: Begin mining using the model-generated solution
    mining_result = start_mining_process(solution)
    if mining_result:
        log_event("[SUCCESS] Mining process executed successfully.")
    else:
        log_event("[FAILURE] Mining process execution failed.")


if __name__ == '__main__':
    main()
