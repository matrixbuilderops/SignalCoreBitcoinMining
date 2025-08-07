"""
Bitcoin mining controller module.

This module handles Bitcoin Core RPC communication, solution validation,
and network submission for the autonomous mining system.
"""

import subprocess  # nosec B404
import json
import time
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
        Dictionary containing RPC response (always returns dict, never None)
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


def get_mempool_info() -> Dict[str, Any]:
    """
    Get current mempool information.

    Returns:
        Dictionary containing mempool status and transaction info
    """
    return call_bitcoin_rpc("getmempoolinfo")


def get_mempool_transactions() -> Dict[str, Any]:
    """
    Get list of transactions in mempool.

    Returns:
        Dictionary containing mempool transactions
    """
    return call_bitcoin_rpc("getrawmempool", [True])


def get_block_height() -> int:
    """
    Get current block height.

    Returns:
        Current block height or 0 if error
    """
    blockchain_info = get_blockchain_info()
    if blockchain_info and blockchain_info.get("result"):
        blocks = blockchain_info["result"].get("blocks", 0)
        return int(blocks) if blocks is not None else 0
    return 0


def get_network_hashrate() -> float:
    """
    Get current network hashrate.

    Returns:
        Network hashrate in hashes per second
    """
    mining_info = get_mining_info()
    if mining_info and mining_info.get("result"):
        hashrate = mining_info["result"].get("networkhashps", 0.0)
        return float(hashrate) if hashrate is not None else 0.0
    return 0.0


def get_difficulty() -> float:
    """
    Get current mining difficulty.

    Returns:
        Current difficulty or 0 if error
    """
    blockchain_info = get_blockchain_info()
    if blockchain_info and blockchain_info.get("result"):
        difficulty = blockchain_info["result"].get("difficulty", 0.0)
        return float(difficulty) if difficulty is not None else 0.0
    return 0.0


def push_transaction(tx_hex: str) -> Optional[str]:
    """
    Push a transaction to the network.

    Args:
        tx_hex: Raw transaction in hexadecimal format

    Returns:
        Transaction ID if successful, None if failed
    """
    result = call_bitcoin_rpc("sendrawtransaction", [tx_hex])

    if result.get("error"):
        print(f"Transaction push failed: {result['error']}")
        return None

    tx_id = result.get("result")
    if tx_id:
        print(f"Transaction pushed successfully: {tx_id}")
        return str(tx_id)
    else:
        print("Transaction push returned no ID")
        return None


def create_mining_transaction(amount: float, fee: float = 0.0001) -> Optional[str]:
    """
    Create a mining transaction for payouts.

    Args:
        amount: Amount to send (in BTC)
        fee: Transaction fee (in BTC)

    Returns:
        Raw transaction hex or None if failed
    """
    try:
        # Create transaction to mining address
        result = call_bitcoin_rpc(
            "createrawtransaction",
            [[], {BITCOIN_ADDRESS: amount}],  # inputs (let wallet choose)  # outputs
        )

        if result.get("error"):
            print(f"Transaction creation failed: {result['error']}")
            return None

        raw_tx = result.get("result")
        if raw_tx:
            # Fund the transaction
            funded_result = call_bitcoin_rpc("fundrawtransaction", [raw_tx])
            if funded_result.get("result"):
                return str(funded_result["result"].get("hex"))

        return None

    except Exception as e:
        print(f"Error creating mining transaction: {e}")
        return None


def validate_wallet_address(address: str) -> bool:
    """
    Validate a Bitcoin wallet address.

    Args:
        address: Bitcoin address to validate

    Returns:
        True if address is valid
    """
    result = call_bitcoin_rpc("validateaddress", [address])
    if result and result.get("result"):
        validation_result = result["result"]
        is_valid = validation_result.get("isvalid", False)
        return bool(is_valid) if is_valid is not None else False
    return False


def get_wallet_balance() -> float:
    """
    Get current wallet balance.

    Returns:
        Wallet balance in BTC
    """
    result = call_bitcoin_rpc("getbalance")
    if result and result.get("result") is not None:
        return float(result["result"])
    return 0.0


def get_chain_data() -> Dict[str, Any]:
    """
    Get comprehensive chain data for mining operations.

    Returns:
        Dictionary with current blockchain state
    """
    blockchain_info = get_blockchain_info()
    mining_info = get_mining_info()
    mempool_info = get_mempool_info()

    chain_data = {
        "block_height": get_block_height(),
        "difficulty": get_difficulty(),
        "network_hashrate": get_network_hashrate(),
        "wallet_balance": get_wallet_balance(),
        "mempool_size": (
            mempool_info.get("result", {}).get("size", 0) if mempool_info else 0
        ),
        "mempool_bytes": (
            mempool_info.get("result", {}).get("bytes", 0) if mempool_info else 0
        ),
        "blockchain_info": blockchain_info.get("result", {}) if blockchain_info else {},
        "mining_info": mining_info.get("result", {}) if mining_info else {},
        "address_valid": validate_wallet_address(BITCOIN_ADDRESS),
        "timestamp": time.time(),
    }

    return chain_data


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
