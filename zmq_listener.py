import zmq
import json
from typing import Callable

class ZMQListener:
    def __init__(self, endpoints: dict[str, str], callback: Callable[[str, bytes], None]) -> None:
        self.endpoints = endpoints
        self.callback = callback
        self.context = zmq.Context()
        self.sockets = []

    def _connect_socket(self, topic: str, address: str) -> None:
        socket = self.context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, '')
        socket.connect(address)
        self.sockets.append((topic, socket))

    def start(self) -> None:
        for topic, address in self.endpoints.items():
            self._connect_socket(topic, address)

        while True:
            for topic, socket in self.sockets:
                try:
                    msg = socket.recv(zmq.NOBLOCK)
                    self.callback(topic, msg)
                except zmq.Again:
                    continue


def default_callback(topic: str, msg: bytes) -> None:
    print(f"[{topic}] Received {len(msg)} bytes")


if __name__ == "__main__":
    endpoints = {
        "hashblock": "tcp://127.0.0.1:28335",
        "rawblock": "tcp://127.0.0.1:28333",
        "hashtx": "tcp://127.0.0.1:28334",
        "rawtx": "tcp://127.0.0.1:28332"
    }

    listener = ZMQListener(endpoints, default_callback)
    listener.start()
