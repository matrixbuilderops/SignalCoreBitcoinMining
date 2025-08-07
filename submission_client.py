"""
submission_client.py — Authenticates and submits block via Bitcoin Core JSON-RPC

Handles Bitcoin Core RPC authentication and block submission.
Integrates with existing mining_controller.py functionality.
"""

import time
from typing import Dict, Any, Optional
from mining_controller import (
    call_bitcoin_rpc,
    submit_solution,
    get_blockchain_info,
    get_mining_info,
    validate_wallet_address,
    get_wallet_balance,
    BITCOIN_ADDRESS,
    BITCOIN_WALLET_NAME,
    BITCOIN_RPC_USER,
)


class SubmissionClient:
    """
    Bitcoin Core RPC client for authentication and block submission.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the submission client.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.submissions_attempted = 0
        self.submissions_successful = 0
        self.last_submission_time = 0

    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] SubmissionClient: {message}")

    def authenticate(self) -> bool:
        """
        Test Bitcoin Core RPC authentication.

        Returns:
            True if authentication successful
        """
        self.log("Testing Bitcoin Core RPC authentication...")

        try:
            # Test basic RPC call
            info = get_blockchain_info()
            if info.get("result"):
                self.log("✓ RPC authentication successful")

                # Validate wallet address
                if validate_wallet_address(BITCOIN_ADDRESS):
                    self.log(f"✓ Wallet address validated: {BITCOIN_ADDRESS}")
                else:
                    self.log(f"⚠ Wallet address validation failed: {BITCOIN_ADDRESS}")

                # Check wallet balance
                balance = get_wallet_balance()
                self.log(f"Wallet balance: {balance} BTC")

                return True
            else:
                self.log(
                    f"✗ RPC authentication failed: {info.get('error', 'Unknown error')}"
                )
                return False

        except Exception as e:
            self.log(f"✗ Authentication error: {str(e)}")
            return False

    def check_network_status(self) -> Dict[str, Any]:
        """
        Check Bitcoin network status and readiness.

        Returns:
            Dictionary with network status information
        """
        self.log("Checking Bitcoin network status...")

        try:
            blockchain_info = get_blockchain_info()
            mining_info = get_mining_info()

            if not blockchain_info.get("result") or not mining_info.get("result"):
                return {
                    "ready": False,
                    "error": "Could not retrieve network information",
                }

            bc_data = blockchain_info["result"]
            mining_data = mining_info["result"]

            status = {
                "ready": True,
                "block_height": bc_data.get("blocks", 0),
                "verification_progress": bc_data.get("verificationprogress", 0),
                "difficulty": bc_data.get("difficulty", 0),
                "network_hash_rate": mining_data.get("networkhashps", 0),
                "chain": bc_data.get("chain", "unknown"),
                "synchronized": bc_data.get("verificationprogress", 0) > 0.99,
            }

            self.log(
                f"Network status: {status['chain']} chain, height {status['block_height']}"
            )
            self.log(f"Synchronization: {status['verification_progress']*100:.1f}%")

            return status

        except Exception as e:
            self.log(f"Error checking network status: {str(e)}")
            return {"ready": False, "error": str(e)}

    def prepare_solution_data(
        self, validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare solution data for submission.

        Args:
            validation_results: Math validation results

        Returns:
            Prepared solution data
        """
        self.log("Preparing solution data for submission...")

        # Convert validation results to submission format
        solution_data = {
            "level": validation_results.get("level", 16000),
            "fork_integrity": validation_results.get("fork_integrity", False),
            "entropy_parity": validation_results.get("entropy_parity", False),
            "fork_sync": validation_results.get("fork_sync", False),
            "sorrell": validation_results.get("sorrell", 0),
            "fork_cluster": validation_results.get("fork_cluster", 0),
            "over_recursion": validation_results.get("over_recursion", 0),
            "bit_load": validation_results.get("bit_load", 0),
            "cycles": validation_results.get("cycles", 0),
            "pre_stabilizer": validation_results.get("pre_stabilizer", ""),
            "post_stabilizer": validation_results.get("post_stabilizer", ""),
            "submission_time": time.time(),
            "wallet_address": BITCOIN_ADDRESS,
            "wallet_name": BITCOIN_WALLET_NAME,
        }

        validation_count = sum(
            [
                solution_data["fork_integrity"],
                solution_data["entropy_parity"],
                solution_data["fork_sync"],
            ]
        )
        self.log(
            f"Solution prepared - Level: {solution_data['level']}, "
            f"Validations: {validation_count}/3"
        )

        return solution_data

    def submit_block_solution(
        self, validation_results: Dict[str, Any]
    ) -> Optional[str]:
        """
        Submit mining solution to Bitcoin network.

        Args:
            validation_results: Validation results from math processing

        Returns:
            Block hash if successful, None if failed
        """
        self.submissions_attempted += 1
        self.log(f"Submitting solution (attempt #{self.submissions_attempted})...")

        try:
            # Prepare solution data
            solution_data = self.prepare_solution_data(validation_results)

            # Record submission time
            self.last_submission_time = time.time()

            # Submit using existing mining controller
            submitted_hash = submit_solution(validation_results)

            if submitted_hash:
                self.submissions_successful += 1
                self.log(f"✓ Solution submitted successfully!")
                self.log(f"  Block Hash: {submitted_hash}")
                self.log(f"  Level: {solution_data['level']}")
                self.log(f"  Submission Rate: {self.get_success_rate():.1f}%")
                return submitted_hash
            else:
                self.log("✗ Solution submission failed validation")
                return None

        except Exception as e:
            self.log(f"✗ Error submitting solution: {str(e)}")
            return None

    def get_success_rate(self) -> float:
        """
        Get submission success rate.

        Returns:
            Success rate as percentage
        """
        if self.submissions_attempted == 0:
            return 0.0
        return (self.submissions_successful / self.submissions_attempted) * 100

    def get_submission_stats(self) -> Dict[str, Any]:
        """
        Get submission statistics.

        Returns:
            Dictionary with submission statistics
        """
        return {
            "total_submissions": self.submissions_attempted,
            "successful_submissions": self.submissions_successful,
            "success_rate_percent": self.get_success_rate(),
            "last_submission_time": self.last_submission_time,
            "wallet_address": BITCOIN_ADDRESS,
            "rpc_user": BITCOIN_RPC_USER,
        }

    def verify_submission_readiness(self) -> bool:
        """
        Verify that the client is ready for submissions.

        Returns:
            True if ready for submissions
        """
        self.log("Verifying submission readiness...")

        # Check authentication
        if not self.authenticate():
            self.log("✗ Authentication check failed")
            return False

        # Check network status
        network_status = self.check_network_status()
        if not network_status.get("ready", False):
            self.log(
                f"✗ Network not ready: {network_status.get('error', 'Unknown error')}"
            )
            return False

        if not network_status.get("synchronized", False):
            self.log("⚠ Network not fully synchronized")
            # Continue anyway, but warn

        self.log("✓ Submission client ready")
        return True

    def emergency_stop(self) -> None:
        """
        Emergency stop for submission client.
        """
        self.log("Emergency stop requested for submission client")
        # Could implement emergency procedures here
        # For now, just log the event


def main():
    """
    Main entry point for testing the submission client.
    """
    print("Testing Bitcoin Core submission client...")

    # Create client
    client = SubmissionClient(verbose=True)

    # Test readiness
    if client.verify_submission_readiness():
        print("✓ Submission client is ready")

        # Test with mock validation data
        test_validation = {
            "level": 16000,
            "fork_integrity": True,
            "entropy_parity": True,
            "fork_sync": True,
            "sorrell": 123456,
            "fork_cluster": 234567,
            "over_recursion": 345678,
            "bit_load": 1600000,
            "cycles": 161,
            "pre_stabilizer": "941d793ce78e45983a4d98d6e4ed0529d923f06f8ecefcabe45c5448c65333fca",
            "post_stabilizer": "74402f56dc3f9154da10ab8d5dbe518db9aa2a332b223bc7bdca9871d0b1a55c",
        }

        print("\nTesting solution preparation...")
        solution_data = client.prepare_solution_data(test_validation)
        print(f"Solution prepared with level: {solution_data['level']}")

        # Show stats
        stats = client.get_submission_stats()
        print(f"\nSubmission Stats: {stats}")

    else:
        print("✗ Submission client not ready")


if __name__ == "__main__":
    main()
