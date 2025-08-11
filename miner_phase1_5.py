import json
import requests
import hashlib
import time
from pathlib import Path

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

RPC_USER = config['rpc_user']
RPC_PASS = config['rpc_password']
RPC_PORT = config['rpc_port']
RPC_HOST = config['rpc_host']
PAYOUT_ADDRESS = config['payout_address']

def rpc_call(method, params=None):
    """Make an RPC call to the Bitcoin node."""
    url = f"http://{RPC_HOST}:{RPC_PORT}"
    headers = {'content-type': 'application/json'}
    payload = json.dumps({
        "method": method,
        "params": params or [],
        "id": int(time.time())
    })
    response = requests.post(url, headers=headers, data=payload, auth=(RPC_USER, RPC_PASS))
    response.raise_for_status()
    return response.json()["result"]

def sha256d(data: bytes) -> bytes:
    """Double SHA-256 hash."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def merkle_root(tx_hashes):
    """Calculate the Merkle root from a list of transaction hashes."""
    if not tx_hashes:
        return None
    hashes = [bytes.fromhex(h)[::-1] for h in tx_hashes]  # reverse byte order
    while len(hashes) > 1:
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])
        new_hashes = []
        for i in range(0, len(hashes), 2):
            new_hashes.append(sha256d(hashes[i] + hashes[i + 1]))
        hashes = new_hashes
    return hashes[0][::-1].hex()  # reverse back

def main():
    print("[*] Fetching block template...")
    template = rpc_call("getblocktemplate", [{"rules": ["segwit"]}])
    print("[+] Got template.")

    print("[*] Building coinbase transaction...")
    coinbase = rpc_call("getnewaddress", ["", "bech32"])
    if PAYOUT_ADDRESS:
        coinbase = PAYOUT_ADDRESS

    # Just grab all tx hashes from template
    tx_hashes = [tx['txid'] for tx in template['transactions']]

    # Add the coinbase transaction's fake hash
    coinbase_txid = "00" * 32  # placeholder
    tx_hashes.insert(0, coinbase_txid)

    print("[*] Calculating Merkle root...")
    root = merkle_root(tx_hashes)
    print(f"[+] Merkle root: {root}")

    block_header = {
        "version": template["version"],
        "previousblockhash": template["previousblockhash"],
        "merkleroot": root,
        "curtime": template["curtime"],
        "bits": template["bits"],
        "height": template["height"]
    }

    output_data = {
        "block_header": block_header,
        "target": template["target"],
        "transactions": tx_hashes
    }

    # Save output for Phase 2
    out_file = Path(__file__).parent / "phase1_5_output.json"
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"[+] Block header + Merkle root saved to {out_file}")

if __name__ == "__main__":
    main()

