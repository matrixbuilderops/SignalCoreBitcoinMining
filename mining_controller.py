import subprocess
import json
from typing import Optional, Dict, Any


# Bitcoin Core RPC configuration from Bitcoin Core Node RPC.txt
BITCOIN_RPC_USER = "SingalCoreBitcoin"
BITCOIN_RPC_PASSWORD = "B1tc0n4L1dz"
BITCOIN_WALLET_NAME = "SignalCoreBitcoinMining"
BITCOIN_ADDRESS = "bc1qcmxyhlpm3lf9zvyevqas9n547lywws2e7wvuu1"


def call_bitcoin_rpc(method: str, params: list = None) -> Dict[str, Any]:
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
        method
    ] + [str(p) for p in params]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return {"result": result.stdout.strip(), "error": None}
        else:
            return {"result": None, "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"result": None, "error": "RPC call timeout"}
    except Exception as e:
        return {"result": None, "error": str(e)}


def get_blockchain_info() -> Dict[str, Any]:
    """Get current blockchain information"""
    return call_bitcoin_rpc("getblockchaininfo")


def get_mining_info() -> Dict[str, Any]:
    """Get current mining information"""
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
        validation_results.get('fork_integrity', False),
        validation_results.get('entropy_parity', False),
        validation_results.get('fork_sync', False)
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
        block_hash = block_hashes[0]
        print(f"Solution submitted successfully. Block hash: {block_hash}")
        return block_hash
    else:
        print("Mining submission returned no block hash")
        return None


def monitor_mining_progress() -> Dict[str, Any]:
    """Monitor current mining progress and network status"""
    blockchain_info = get_blockchain_info()
    mining_info = get_mining_info()
    
    return {
        "blockchain_info": blockchain_info,
        "mining_info": mining_info,
        "network_active": blockchain_info.get("result", {}).get("verificationprogress", 0) > 0.99
    }

