import threading
import time
import sys


class SignalCoreMonitor:
    def __init__(self, silent_mode: bool = True):
        self.silent_mode = silent_mode
        self._running = False
        self._spinner_thread = None
        self._status_message = "Initializing"

    def _spinner(self):
        spinner_sequence = ['|', '/', '-', '\\']
        idx = 0
        while self._running:
            if not self.silent_mode:
                sys.stdout.write(f"\r{self._status_message} {spinner_sequence[idx % len(spinner_sequence)]}")
                sys.stdout.flush()
                idx += 1
            time.sleep(0.1)

    def start(self, message: str = "Running"):
        self._status_message = message
        self._running = True
        if self._spinner_thread is None or not self._spinner_thread.is_alive():
            self._spinner_thread = threading.Thread(target=self._spinner, daemon=True)
            self._spinner_thread.start()

    def update_message(self, message: str):
        self._status_message = message

    def stop(self):
        self._running = False
        if not self.silent_mode:
            sys.stdout.write("\rDone.\n")
            sys.stdout.flush()


# Example usage:
if __name__ == "__main__":
    monitor = SignalCoreMonitor(silent_mode=False)
    monitor.start("Mining")
    try:
        time.sleep(5)  # Simulate mining
    finally:
        monitor.stop()
