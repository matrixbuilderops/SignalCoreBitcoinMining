import subprocess

def submit_solution():
    cmd = ["bitcoin-cli", "generatetoaddress", "1", "<YOUR_BITCOIN_ADDRESS>"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    txid = result.stdout.strip()
    print(f"Solution submitted. Block hash: {txid}")
    return txid

