import logging
from zmq_listener import ZMQListener
from model_interface import RecursionModel
from mining_engine import MiningEngine
from rpc_client import RPCClient
from utils import load_config, validate_environment


def main():
    config = load_config("config.json")
    logging.basicConfig(level=logging.INFO)

    validate_environment(config)

    zmq_listener = ZMQListener(config["zmq"])
    rpc_client = RPCClient(config["rpc"])
    model = RecursionModel(config["model"])
    miner = MiningEngine(model=model, rpc_client=rpc_client)

    logging.info("System initialized. Awaiting blockchain events...")

    for event in zmq_listener.listen():
        if event["type"] == "rawblock":
            block_data = event["data"]
            logging.info("New block detected, initializing model sequence.")
            candidate_block = rpc_client.get_block_template()
            equation = model.generate_equation(level=16000)
            solution = model.solve_equation(equation, block_data)
            miner.submit_solution(solution, candidate_block)


if __name__ == "__main__":
    main()
