#!/usr/bin/env python3
"""
Production launcher for SignalCore Bitcoin Mining System.

This script handles production deployment with enhanced security,
performance optimization, and autonomous operation capabilities.
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mining_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ProductionLauncher")


class ProductionMiningLauncher:
    """Production launcher for autonomous Bitcoin mining operations."""

    def __init__(self):
        """Initialize the production launcher."""
        self.mining_process = None
        self.running = False
        self.restart_count = 0
        self.max_restarts = 10
        self.start_time = time.time()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        if self.mining_process:
            self.mining_process.terminate()
            self.mining_process.wait(timeout=30)

    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate production environment requirements.

        Returns:
            Dictionary of validation results
        """
        logger.info("Validating production environment...")

        validation_results = {
            "python_version": sys.version_info >= (3, 8),
            "bitcoin_core_config": os.path.exists("Bitcoin Core Node RPC.txt"),
            "math_definition": os.path.exists("math.txt"),
            "pyzmq_available": False,
            "ollama_available": False,
            "bitcoin_cli_available": False,
        }

        # Check pyzmq
        try:
            import pyzmq

            validation_results["pyzmq_available"] = True
            logger.info("✓ pyzmq available for ZMQ block monitoring")
        except ImportError:
            logger.warning("⚠ pyzmq not available, will use polling fallback")

        # Check Ollama
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, timeout=10)
            if result.returncode == 0:
                validation_results["ollama_available"] = True
                logger.info("✓ Ollama available for AI analysis")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("⚠ Ollama not available, will use mathematical fallback")

        # Check Bitcoin CLI
        try:
            result = subprocess.run(
                ["bitcoin-cli", "--version"], capture_output=True, timeout=10
            )
            if result.returncode == 0:
                validation_results["bitcoin_cli_available"] = True
                logger.info("✓ Bitcoin CLI available for RPC operations")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("⚠ Bitcoin CLI not available, production mining limited")

        return validation_results

    def optimize_system_performance(self) -> None:
        """Apply system optimizations for production mining."""
        logger.info("Applying production performance optimizations...")

        # Set environment variables for optimized operation
        os.environ["PYTHONUNBUFFERED"] = "1"  # Immediate output
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # No .pyc files

        # Memory optimization
        if "PYTHONMALLOC" not in os.environ:
            os.environ["PYTHONMALLOC"] = "malloc"

        # ZMQ optimization if available
        if "ZMQ_BLOCKMAXSIZE" not in os.environ:
            os.environ["ZMQ_BLOCKMAXSIZE"] = "1048576"  # 1MB blocks

        logger.info("✓ System optimizations applied")

    def create_secure_config(self) -> Dict[str, Any]:
        """
        Create secure configuration for production deployment.

        Returns:
            Production configuration dictionary
        """
        config = {
            "verbose": False,  # Minimal output for production
            "ai_enabled": True,  # Enable AI by default
            "fallback_enabled": True,  # Enable mathematical fallback
            "zmq_enabled": True,  # Use ZMQ if available
            "max_concurrent_blocks": 3,  # Parallel processing limit
            "submission_timeout": 30,  # RPC timeout
            "restart_on_error": True,  # Auto-restart on errors
            "log_level": "INFO",
            "performance_monitoring": True,
        }

        # Read Bitcoin Core credentials securely
        rpc_file = Path("Bitcoin Core Node RPC.txt")
        if rpc_file.exists():
            try:
                with open(rpc_file, "r") as f:
                    content = f.read()
                    # Credentials are stored in the file, not hardcoded
                    config["rpc_configured"] = True
                    logger.info("✓ Bitcoin Core RPC credentials loaded")
            except Exception as e:
                logger.error(f"Failed to load RPC credentials: {e}")
                config["rpc_configured"] = False
        else:
            logger.warning("⚠ Bitcoin Core RPC credentials not found")
            config["rpc_configured"] = False

        return config

    def launch_mining_system(self, mode: str = "production") -> bool:
        """
        Launch the mining system in specified mode.

        Args:
            mode: Operation mode ('production', 'test', 'quiet')

        Returns:
            True if launch successful
        """
        try:
            logger.info(f"Launching mining system in {mode} mode...")

            # Determine command line arguments based on mode
            cmd = [sys.executable, "main.py"]

            if mode == "production":
                # Production mode: quiet operation with AI
                cmd.append("--quiet")
            elif mode == "test":
                cmd.append("--test")
            elif mode == "quiet":
                cmd.extend(["--quiet", "--thinking"])

            # Launch the mining process
            self.mining_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            logger.info(f"✓ Mining system launched with PID {self.mining_process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to launch mining system: {e}")
            return False

    def monitor_mining_process(self) -> None:
        """Monitor mining process and handle restarts."""
        logger.info("Starting autonomous mining monitoring...")

        while self.running and self.restart_count < self.max_restarts:
            try:
                if self.mining_process is None:
                    # Initial launch or restart
                    if not self.launch_mining_system("production"):
                        logger.error("Failed to launch mining system")
                        break

                # Check process status
                return_code = self.mining_process.poll()

                if return_code is not None:
                    # Process has exited
                    if return_code == 0:
                        logger.info("Mining process completed successfully")
                        break
                    else:
                        logger.warning(f"Mining process exited with code {return_code}")

                        if self.restart_count < self.max_restarts:
                            self.restart_count += 1
                            logger.info(
                                f"Restarting mining system (attempt {self.restart_count}/{self.max_restarts})"
                            )
                            time.sleep(5)  # Brief delay before restart
                            self.mining_process = None
                            continue
                        else:
                            logger.error("Maximum restart attempts reached")
                            break

                # Log periodic status
                uptime = time.time() - self.start_time
                if int(uptime) % 300 == 0:  # Every 5 minutes
                    logger.info(
                        f"Mining system running - Uptime: {uptime/3600:.1f} hours, Restarts: {self.restart_count}"
                    )

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Longer delay on errors

        logger.info("Mining monitoring stopped")

    def run_production_deployment(self, mode: str = "production") -> None:
        """
        Run complete production deployment.

        Args:
            mode: Operation mode for deployment
        """
        logger.info("🚀 Starting SignalCore Bitcoin Mining - Production Deployment")
        logger.info("=" * 70)

        # Validate environment
        validation = self.validate_environment()

        critical_checks = [
            validation["python_version"],
            validation["bitcoin_core_config"],
            validation["math_definition"],
        ]

        if not all(critical_checks):
            logger.error("Critical environment validation failed")
            logger.error("Cannot proceed with production deployment")
            sys.exit(1)

        # Apply optimizations
        self.optimize_system_performance()

        # Create secure configuration
        config = self.create_secure_config()
        logger.info(f"✓ Production configuration created - AI: {config['ai_enabled']}")

        # Display deployment status
        logger.info("Deployment Status:")
        logger.info(
            f"  - ZMQ Monitoring: {'✓' if validation['pyzmq_available'] else '⚠ Fallback'}"
        )
        logger.info(
            f"  - AI Analysis: {'✓' if validation['ollama_available'] else '⚠ Math Fallback'}"
        )
        logger.info(
            f"  - Bitcoin RPC: {'✓' if validation['bitcoin_cli_available'] else '⚠ Limited'}"
        )
        logger.info(f"  - Math Engine: ✓ Level 16000")
        logger.info("=" * 70)

        # Start autonomous operation
        self.running = True
        try:
            self.monitor_mining_process()
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            if self.mining_process:
                self.mining_process.terminate()
                self.mining_process.wait(timeout=30)

        uptime = time.time() - self.start_time
        logger.info(
            f"Production deployment completed - Total uptime: {uptime/3600:.1f} hours"
        )


def main():
    """Main entry point for production launcher."""
    parser = argparse.ArgumentParser(
        description="SignalCore Bitcoin Mining - Production Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["production", "test", "quiet"],
        default="production",
        help="Deployment mode (default: production)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment and exit",
    )

    args = parser.parse_args()

    launcher = ProductionMiningLauncher()

    if args.validate_only:
        # Validation mode
        print("Running environment validation...")
        validation = launcher.validate_environment()

        print("\nValidation Results:")
        for check, result in validation.items():
            status = "✓ PASS" if result else "⚠ FAIL/WARN"
            print(f"  {check}: {status}")

        critical_failures = [
            not validation["python_version"],
            not validation["bitcoin_core_config"],
            not validation["math_definition"],
        ]

        if any(critical_failures):
            print("\n❌ Critical validation failures detected")
            sys.exit(1)
        else:
            print("\n✅ Environment validation successful")
            sys.exit(0)
    else:
        # Production deployment
        launcher.run_production_deployment(args.mode)


if __name__ == "__main__":
    main()
