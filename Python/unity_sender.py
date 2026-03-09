import zmq
import json
import time


class UnitySender:
    def __init__(self, port=5555):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, 50)
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait when closing
        self.socket.bind(f"tcp://*:{port}")
        self.last_error_time = 0

    def send(self, first_letter, first_conf, second_letter, second_conf):
        try:
            message = {
                "first": str(first_letter),
                "first_conf": float(first_conf),
                "second": str(second_letter),
                "second_conf": float(second_conf)
            }
            # Send like your working code - no NOBLOCK flag
            self.socket.send_string(json.dumps(message))
            return True

        except Exception as e:
            # This should rarely happen now
            return False


# if __name__ == "__main__":
#     unity = UnitySender()
#     unity.send("A", 0.85, "B", 0.12)