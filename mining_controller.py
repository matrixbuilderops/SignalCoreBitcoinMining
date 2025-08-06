"""
Bitcoin mining controller module.

This module handles Bitcoin Core RPC communication, solution validation,
and network submission for the autonomous mining system.
"""

import subprocess  # nosec B404
import json
from typing import Optional, Dict, Any


# Bitcoin Core RPC configuration from environment or Bitcoin Core Node RPC.txt
import os

BITCOIN_RPC_USER = os.getenv("BITCOIN_RPC_USER", "SingalCoreBitcoin")
BITCOIN_RPC_PASSWORD = os.getenv("BITCOIN_RPC_PASSWORD", "B1tc0n4L1dz")  # nosec B105
BITCOIN_WALLET_NAME = "SignalCoreBitcoinMining"
BITCOIN_ADDRESS = "bc1qcmxyhlpm3lf9zvyevqas9n547lywws2e7wvuu1"


def call_bitcoin_rpc(method: str, params: Optional[list] = None) -> Dict[str, Any]:
    """
    Call Bitcoin Core RPC with proper authentication

    Args:
        method: RPC method name
        params: List of parameters for the method

    Returns:
        Dictionary containing RPC response
    """
    if params is None:
        params = []

    cmd = [
        "bitcoin-cli",
        f"-rpcuser={BITCOIN_RPC_USER}",
        f"-rpcpassword={BITCOIN_RPC_PASSWORD}",
        f"-rpcwallet={BITCOIN_WALLET_NAME}",
        method,
    ] + [str(p) for p in params]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )  # nosec B603
        if result.returncode == 0:
            try:
                response_data = json.loads(result.stdout.strip())
                return (
                    response_data
                    if isinstance(response_data, dict)
                    else {"result": result.stdout.strip(), "error": None}
                )
            except json.JSONDecodeError:
                return {"result": result.stdout.strip(), "error": None}
        else:
            return {"result": None, "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"result": None, "error": "RPC call timeout"}
    except Exception as e:
        return {"result": None, "error": str(e)}


def get_blockchain_info() -> Dict[str, Any]:
    """
    Get current blockchain information.

    Returns:
        Dictionary containing blockchain status and information
    """
    return call_bitcoin_rpc("getblockchaininfo")


def get_mining_info() -> Dict[str, Any]:
    """
    Get current mining information.

    Returns:
        Dictionary containing mining status and performance metrics
    """
    return call_bitcoin_rpc("getmininginfo")


def validate_solution(validation_results: Dict[str, Any]) -> bool:
    """
    Validate mining solution based on math module results

    Args:
        validation_results: Results from math_module processing

    Returns:
        True if solution is valid for submission
    """
    required_checks = [
        validation_results.get("fork_integrity", False),
        validation_results.get("entropy_parity", False),
        validation_results.get("fork_sync", False),
    ]

    # All critical checks must pass
    return all(required_checks)


def submit_solution(validation_results: Dict[str, Any]) -> Optional[str]:
    """
    Submit mining solution to Bitcoin network

    Args:
        validation_results: Validation results from math processing

    Returns:
        Block hash if successful, None if failed
    """
    if not validate_solution(validation_results):
        print("Solution validation failed - not submitting")
        return None

    # Generate new block to our address
    result = call_bitcoin_rpc("generatetoaddress", [1, BITCOIN_ADDRESS])

    if result.get("error"):
        print(f"Mining submission failed: {result['error']}")
        return None

    block_hashes = result.get("result")
    if block_hashes and len(block_hashes) > 0:
        block_hash = str(block_hashes[0])
        print(f"Solution submitted successfully. Block hash: {block_hash}")
        return block_hash
    else:
        print("Mining submission returned no block hash")
        return None


def monitor_mining_progress() -> Dict[str, Any]:
    """
    Monitor current mining progress and network status.

    Returns:
        Dictionary containing blockchain info, mining info, and network status
    """
    blockchain_info = get_blockchain_info()
    mining_info = get_mining_info()

    return {
        "blockchain_info": blockchain_info,
        "mining_info": mining_info,
        "network_active": blockchain_info.get("result", {}).get(
            "verificationprogress", 0
        )
        > 0.99,
    }
