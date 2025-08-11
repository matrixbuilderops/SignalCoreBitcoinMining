# pre_mine_test.py
import json
import subprocess
from pathlib import Path

# === Config ===
RPCWALLET = "SignalCoreBitcoinMining"
OUTPUT_FILE = Path("pre_mine_test_output.json")

def run_cli(args):
    """Run bitcoin-cli with given args and return parsed JSON if possible."""
    cmd = ["bitcoin-cli", f"-rpcwallet={RPCWALLET}"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    result.check_returncode()
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return result.stdout.strip()

def main():
    print("[*] Fetching block template...")
    block_template = run_cli(["getblocktemplate", '{"rules":["segwit"]}'])

    # Get the coinbase transaction info
    coinbase_tx = None
    if "coinbaseaux" in block_template:
        coinbase_tx = block_template["coinbaseaux"]
    else:
        coinbase_tx = "Coinbase details not directly in template."

    # Merkle root — in Bitcoin, you build this from transactions
    # For now, we'll grab the tx list; Merkle root calculation can be added later
    tx_list = block_template.get("transactions", [])
    merkle_root_placeholder = "Merkle root would be computed from tx list"

    output_data = {
        "block_template": block_template,
        "coinbase_tx_info": coinbase_tx,
        "transaction_count": len(tx_list),
        "merkle_root": merkle_root_placeholder
    }

    # Save to file
    OUTPUT_FILE.write_text(json.dumps(output_data, indent=2))
    print(f"[+] Results saved to {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()

