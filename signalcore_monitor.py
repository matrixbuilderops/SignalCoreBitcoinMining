import threading
import time
import sys
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum


class MonitorState(Enum):
    """Monitor state enumeration"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class MonitorStats:
    """Monitor statistics"""
    start_time: float
    messages_displayed: int = 0
    errors_logged: int = 0
    warnings_logged: int = 0
    uptime_seconds: float = 0.0


class SignalCoreMonitor:
    """Enhanced SignalCore Monitor with error resilience and better silent mode handling"""
    
    def __init__(self, silent_mode: bool = True, log_file: Optional[str] = None):
        self.silent_mode = silent_mode
        self.log_file = log_file
        self._running = False
        self._spinner_thread = None
        self._status_message = "Initializing"
        self._state = MonitorState.STOPPED
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._stats = MonitorStats(start_time=time.time())
        
        # Setup logging
        self._setup_logging()
        
        # Error recovery settings
        self._max_retries = 3
        self._retry_count = 0
        self._last_error: Optional[Exception] = None

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = logging.WARNING if self.silent_mode else logging.INFO
        
        if self.log_file:
            logging.basicConfig(
                filename=self.log_file,
                level=log_level,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        else:
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def _spinner(self):
        """Enhanced spinner with error handling and crash resilience"""
        spinner_sequence = ['|', '/', '-', '\\']
        idx = 0
        
        try:
            while self._running and self._state == MonitorState.RUNNING:
                try:
                    if not self.silent_mode:
                        sys.stdout.write(f"\r{self._status_message} {spinner_sequence[idx % len(spinner_sequence)]}")
                        sys.stdout.flush()
                        self._stats.messages_displayed += 1
                        idx += 1
                    
                    time.sleep(0.1)
                    
                except (BrokenPipeError, IOError) as e:
                    # Handle terminal disconnection gracefully
                    logging.warning(f"Terminal output error: {e}")
                    self._stats.errors_logged += 1
                    if not self.silent_mode:
                        self.silent_mode = True  # Switch to silent mode on terminal issues
                    time.sleep(1)  # Longer sleep when output fails
                    
                except Exception as e:
                    self._handle_spinner_error(e)
                    break
                    
        except Exception as e:
            self._handle_critical_error(e)
        finally:
            self._state = MonitorState.STOPPED

    def _handle_spinner_error(self, error: Exception) -> None:
        """Handle spinner-specific errors with retry logic"""
        self._last_error = error
        self._retry_count += 1
        self._stats.errors_logged += 1
        
        logging.error(f"Spinner error (attempt {self._retry_count}): {error}")
        
        if self._retry_count < self._max_retries:
            time.sleep(2 ** self._retry_count)  # Exponential backoff
            logging.info(f"Retrying spinner (attempt {self._retry_count + 1}/{self._max_retries})")
        else:
            logging.error(f"Spinner failed after {self._max_retries} attempts, switching to silent mode")
            self.silent_mode = True
            self._state = MonitorState.ERROR

    def _handle_critical_error(self, error: Exception) -> None:
        """Handle critical errors that may crash the monitor"""
        self._last_error = error
        self._state = MonitorState.ERROR
        self._stats.errors_logged += 1
        
        logging.critical(f"Critical monitor error: {error}")
        
        if self._error_callback:
            try:
                self._error_callback(error)
            except Exception as callback_error:
                logging.error(f"Error callback failed: {callback_error}")

    def start(self, message: str = "Running", error_callback: Optional[Callable[[Exception], None]] = None):
        """Start monitor with enhanced error handling"""
        self._status_message = message
        self._error_callback = error_callback
        self._running = True
        self._state = MonitorState.STARTING
        self._retry_count = 0  # Reset retry count
        self._stats.start_time = time.time()
        
        logging.info(f"Starting monitor: {message}")
        
        try:
            if self._spinner_thread is None or not self._spinner_thread.is_alive():
                self._state = MonitorState.RUNNING
                self._spinner_thread = threading.Thread(target=self._spinner, daemon=True)
                self._spinner_thread.start()
                
        except Exception as e:
            self._handle_critical_error(e)

    def update_message(self, message: str):
        """Update status message with validation"""
        try:
            if len(message) > 100:  # Prevent excessively long messages
                message = message[:97] + "..."
                self._stats.warnings_logged += 1
                logging.warning("Status message truncated (too long)")
                
            self._status_message = message
            logging.debug(f"Status updated: {message}")
            
        except Exception as e:
            logging.error(f"Failed to update message: {e}")
            self._stats.errors_logged += 1

    def stop(self, final_message: str = "Done."):
        """Stop monitor gracefully with proper cleanup"""
        self._state = MonitorState.STOPPING
        self._running = False
        
        try:
            # Wait for spinner thread to finish (with timeout)
            if self._spinner_thread and self._spinner_thread.is_alive():
                self._spinner_thread.join(timeout=2.0)
            
            if not self.silent_mode:
                sys.stdout.write(f"\r{final_message}\n")
                sys.stdout.flush()
            
            # Update stats
            self._stats.uptime_seconds = time.time() - self._stats.start_time
            
            logging.info(f"Monitor stopped: {final_message}")
            logging.info(f"Monitor stats: {self.get_stats()}")
            
        except Exception as e:
            logging.error(f"Error during monitor shutdown: {e}")
        finally:
            self._state = MonitorState.STOPPED

    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics"""
        current_time = time.time()
        uptime = current_time - self._stats.start_time
        
        return {
            "state": self._state.value,
            "uptime_seconds": uptime,
            "messages_displayed": self._stats.messages_displayed,
            "errors_logged": self._stats.errors_logged,
            "warnings_logged": self._stats.warnings_logged,
            "retry_count": self._retry_count,
            "silent_mode": self.silent_mode,
            "last_error": str(self._last_error) if self._last_error else None
        }

    def health_check(self) -> bool:
        """Perform health check on monitor"""
        try:
            if self._state == MonitorState.ERROR:
                return False
            
            if self._running and self._state == MonitorState.RUNNING:
                # Check if spinner thread is alive
                if self._spinner_thread:
                    return self._spinner_thread.is_alive()
            
            return self._state in [MonitorState.STOPPED, MonitorState.RUNNING]
            
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False

    def force_silent_mode(self) -> None:
        """Force silent mode (useful for service/daemon mode)"""
        self.silent_mode = True
        logging.info("Monitor switched to forced silent mode")

    def is_running(self) -> bool:
        """Check if monitor is currently running"""
        return self._running and self._state == MonitorState.RUNNING


def create_process_monitor(silent: bool = True, log_file: Optional[str] = None) -> SignalCoreMonitor:
    """Factory function to create a monitor for process supervision"""
    return SignalCoreMonitor(silent_mode=silent, log_file=log_file)


# Example usage:
if __name__ == "__main__":
    def error_handler(error: Exception):
        print(f"Custom error handler called: {error}")
    
    monitor = SignalCoreMonitor(silent_mode=False)
    monitor.start("Mining", error_callback=error_handler)
    
    try:
        for i in range(10):
            monitor.update_message(f"Processing block {i+1}")
            time.sleep(1)
            
            # Simulate health check
            if not monitor.health_check():
                print("Health check failed!")
                break
                
    finally:
        stats = monitor.get_stats()
        print(f"Final stats: {stats}")
        monitor.stop()
