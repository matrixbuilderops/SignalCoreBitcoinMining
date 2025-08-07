#!/usr/bin/env python3
"""
Process supervision and daemon management for SignalCore Bitcoin Mining System.
Provides robust process management, supervision, and service capabilities.
"""

import os
import sys
import time
import signal
import logging
import threading
import subprocess
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class ProcessState(Enum):
    """Process state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    CRASHED = "crashed"
    UNKNOWN = "unknown"


@dataclass
class ProcessStats:
    """Process statistics and health information"""
    start_time: float
    restart_count: int = 0
    crash_count: int = 0
    uptime_seconds: float = 0.0
    last_restart: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0


class ProcessSupervisor:
    """Process supervisor with automatic restart and health monitoring"""
    
    def __init__(self, 
                 name: str,
                 command: list,
                 working_dir: Optional[str] = None,
                 max_restarts: int = 5,
                 restart_delay: int = 5,
                 log_file: Optional[str] = None):
        self.name = name
        self.command = command
        self.working_dir = working_dir or os.getcwd()
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        
        # State management
        self.state = ProcessState.STOPPED
        self.process: Optional[subprocess.Popen] = None
        self.stats = ProcessStats(start_time=time.time())
        self._stop_requested = False
        self._supervisor_thread: Optional[threading.Thread] = None
        
        # Logging setup
        self.log_file = log_file
        self._setup_logging()
        
        # Callbacks
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None
        self.on_crash: Optional[Callable[[Exception], None]] = None
        self.on_restart: Optional[Callable] = None

    def _setup_logging(self):
        """Setup logging for the supervisor"""
        self.logger = logging.getLogger(f"supervisor_{self.name}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            if self.log_file:
                handler = logging.FileHandler(self.log_file)
            else:
                handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def start(self) -> bool:
        """Start the supervised process"""
        if self.state in [ProcessState.STARTING, ProcessState.RUNNING]:
            self.logger.warning(f"Process {self.name} is already running")
            return True
        
        try:
            self.state = ProcessState.STARTING
            self._stop_requested = False
            
            # Start supervisor thread
            self._supervisor_thread = threading.Thread(
                target=self._supervisor_loop,
                daemon=True
            )
            self._supervisor_thread.start()
            
            self.logger.info(f"Supervisor started for {self.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start supervisor: {e}")
            self.state = ProcessState.CRASHED
            return False

    def _supervisor_loop(self):
        """Main supervisor loop with restart logic"""
        while not self._stop_requested:
            try:
                if self._should_start_process():
                    self._start_process()
                
                if self.process and self.state == ProcessState.RUNNING:
                    # Monitor process health
                    if self.process.poll() is not None:
                        # Process has exited
                        self._handle_process_exit()
                    else:
                        # Update statistics
                        self._update_stats()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Supervisor loop error: {e}")
                if self.on_crash:
                    self.on_crash(e)
                time.sleep(self.restart_delay)

    def _should_start_process(self) -> bool:
        """Determine if process should be started"""
        if self._stop_requested:
            return False
        
        if self.state == ProcessState.STOPPED:
            return True
        
        if self.state == ProcessState.CRASHED:
            if self.stats.restart_count < self.max_restarts:
                # Check restart delay
                if (self.stats.last_restart is None or 
                    time.time() - self.stats.last_restart >= self.restart_delay):
                    return True
        
        return False

    def _start_process(self):
        """Start the actual process"""
        try:
            self.logger.info(f"Starting process {self.name}: {' '.join(self.command)}")
            
            self.process = subprocess.Popen(
                self.command,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.state = ProcessState.RUNNING
            self.stats.start_time = time.time()
            
            if self.stats.restart_count > 0:
                self.stats.last_restart = time.time()
                if self.on_restart:
                    self.on_restart()
            
            if self.on_start:
                self.on_start()
            
            self.logger.info(f"Process {self.name} started with PID {self.process.pid}")
            
        except Exception as e:
            self.logger.error(f"Failed to start process {self.name}: {e}")
            self.state = ProcessState.CRASHED
            self.stats.crash_count += 1

    def _handle_process_exit(self):
        """Handle process exit"""
        exit_code = self.process.returncode
        self.logger.info(f"Process {self.name} exited with code {exit_code}")
        
        if exit_code == 0:
            # Clean exit
            self.state = ProcessState.STOPPED
        else:
            # Unexpected exit
            self.state = ProcessState.CRASHED
            self.stats.crash_count += 1
            self.stats.restart_count += 1
            
            self.logger.warning(
                f"Process {self.name} crashed (exit code {exit_code}), "
                f"restart attempt {self.stats.restart_count}/{self.max_restarts}"
            )

    def _update_stats(self):
        """Update process statistics"""
        if self.process:
            self.stats.uptime_seconds = time.time() - self.stats.start_time
            
            # Try to get memory and CPU info (requires psutil, optional)
            try:
                import psutil
                proc = psutil.Process(self.process.pid)
                self.stats.memory_usage_mb = proc.memory_info().rss / 1024 / 1024
                self.stats.cpu_percent = proc.cpu_percent()
            except (ImportError, psutil.NoSuchProcess):
                pass

    def stop(self, timeout: int = 10) -> bool:
        """Stop the supervised process gracefully"""
        self.logger.info(f"Stopping process {self.name}")
        self._stop_requested = True
        self.state = ProcessState.STOPPING
        
        if self.process:
            try:
                # Try graceful shutdown first
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.logger.warning(f"Force killing process {self.name}")
                    self.process.kill()
                    self.process.wait()
                
                self.logger.info(f"Process {self.name} stopped")
                
            except Exception as e:
                self.logger.error(f"Error stopping process {self.name}: {e}")
                return False
            finally:
                self.process = None
        
        self.state = ProcessState.STOPPED
        
        if self.on_stop:
            self.on_stop()
        
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current process status and statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "pid": self.process.pid if self.process else None,
            "command": self.command,
            "working_dir": self.working_dir,
            "uptime_seconds": self.stats.uptime_seconds,
            "restart_count": self.stats.restart_count,
            "crash_count": self.stats.crash_count,
            "memory_usage_mb": self.stats.memory_usage_mb,
            "cpu_percent": self.stats.cpu_percent,
            "max_restarts": self.max_restarts,
            "restart_delay": self.restart_delay
        }

    def is_healthy(self) -> bool:
        """Check if process is healthy"""
        return (self.state == ProcessState.RUNNING and 
                self.process is not None and 
                self.process.poll() is None and
                self.stats.crash_count < self.max_restarts)


class ServiceManager:
    """Service manager for managing multiple supervised processes"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.supervisors: Dict[str, ProcessSupervisor] = {}
        self.config_file = config_file
        self._signal_handlers_installed = False
        
        # Setup logging
        self.logger = logging.getLogger("service_manager")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def add_service(self, supervisor: ProcessSupervisor) -> bool:
        """Add a service to be managed"""
        if supervisor.name in self.supervisors:
            self.logger.warning(f"Service {supervisor.name} already exists")
            return False
        
        self.supervisors[supervisor.name] = supervisor
        self.logger.info(f"Added service: {supervisor.name}")
        return True

    def remove_service(self, name: str) -> bool:
        """Remove a service from management"""
        if name not in self.supervisors:
            self.logger.warning(f"Service {name} not found")
            return False
        
        supervisor = self.supervisors[name]
        supervisor.stop()
        del self.supervisors[name]
        self.logger.info(f"Removed service: {name}")
        return True

    def start_service(self, name: str) -> bool:
        """Start a specific service"""
        if name not in self.supervisors:
            self.logger.error(f"Service {name} not found")
            return False
        
        return self.supervisors[name].start()

    def stop_service(self, name: str) -> bool:
        """Stop a specific service"""
        if name not in self.supervisors:
            self.logger.error(f"Service {name} not found")
            return False
        
        return self.supervisors[name].stop()

    def start_all(self) -> bool:
        """Start all managed services"""
        self.logger.info("Starting all services")
        success = True
        
        for name, supervisor in self.supervisors.items():
            if not supervisor.start():
                self.logger.error(f"Failed to start service: {name}")
                success = False
        
        return success

    def stop_all(self) -> bool:
        """Stop all managed services"""
        self.logger.info("Stopping all services")
        success = True
        
        for name, supervisor in self.supervisors.items():
            if not supervisor.stop():
                self.logger.error(f"Failed to stop service: {name}")
                success = False
        
        return success

    def get_status_all(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services"""
        return {name: supervisor.get_status() 
                for name, supervisor in self.supervisors.items()}

    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown"""
        if self._signal_handlers_installed:
            return
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully")
            self.stop_all()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        self._signal_handlers_installed = True

    def run_as_daemon(self):
        """Run service manager as a daemon"""
        self.logger.info("Running as daemon")
        self.install_signal_handlers()
        
        try:
            while True:
                # Health check all services
                for name, supervisor in self.supervisors.items():
                    if not supervisor.is_healthy():
                        self.logger.warning(f"Service {name} is unhealthy")
                
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            self.logger.info("Daemon interrupted, shutting down")
            self.stop_all()


def create_mining_service_manager() -> ServiceManager:
    """Create a service manager configured for Bitcoin mining"""
    manager = ServiceManager()
    
    # Create supervisor for the main mining process
    mining_supervisor = ProcessSupervisor(
        name="bitcoin_mining_core",
        command=[sys.executable, "bitcoin_mining_core.py"],
        max_restarts=10,
        restart_delay=30,
        log_file="logs/mining_core.log"
    )
    
    # Add callbacks for mining events
    def on_mining_start():
        logging.info("Bitcoin mining core started")
    
    def on_mining_crash(error):
        logging.error(f"Bitcoin mining core crashed: {error}")
    
    mining_supervisor.on_start = on_mining_start
    mining_supervisor.on_crash = on_mining_crash
    
    manager.add_service(mining_supervisor)
    
    return manager


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SignalCore Process Manager")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--start", help="Start specific service")
    parser.add_argument("--stop", help="Stop specific service")
    parser.add_argument("--status", action="store_true", help="Show status of all services")
    
    args = parser.parse_args()
    
    # Create mining service manager
    manager = create_mining_service_manager()
    
    if args.daemon:
        manager.start_all()
        manager.run_as_daemon()
    elif args.start:
        success = manager.start_service(args.start)
        sys.exit(0 if success else 1)
    elif args.stop:
        success = manager.stop_service(args.stop)
        sys.exit(0 if success else 1)
    elif args.status:
        status = manager.get_status_all()
        print(json.dumps(status, indent=2))
    else:
        print("Use --help for usage information")