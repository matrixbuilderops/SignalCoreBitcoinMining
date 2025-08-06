import zmq

def listen_for_blocks(callback):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:28332")  # rawtx port example
    socket.setsockopt_string(zmq.SUBSCRIBE, "hashblock")
    socket.setsockopt_string(zmq.SUBSCRIBE, "rawblock")

    while True:
        topic, message = socket.recv_multipart()
        callback(topic, message)

