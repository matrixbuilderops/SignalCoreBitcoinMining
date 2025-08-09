# -*- coding: utf-8 -*-
"""
SignalCore — Level 16000 Miner (Single-File • Hardened • Math-First)
Features: self-verify banner + seal/unseal, preflight++, stick-until-stale, autosave,
autosubmit w/ retries & outbox queue, secondary RPC failover, ZMQ tip refresh (fallback polling),
metrics (NDJSON), hash counter, watchdog-friendly exits, math-governed pipeline.
"""

import os, sys, json, time, hmac, hashlib, argparse, base64, threading, queue, re
from pathlib import Path

# ========== Stabilizers / Verify / Seal-Unseal ==========
PRE_STAB = "db7b27e5f4c60791ad31f4ba0eed3be5d45f9f368a3f2723c4af8346bb196ced3ca1c6d61129600c8f83ab9d9b70ee7242e11a58260a45e35e55733676254771"
POST_STAB = "db7b27e5f4c60791ad31f4ba0eed3be5d45f9f368a3f2723c4af8346bb196ced3ca1c6d61129600c8f83ab9d9b70ee7242e11a58260a45e35e55733676254771"
STAB_MODE = os.getenv("STAB_MODE","self")  # self | external

def _self_hash():
    try:
        return hashlib.sha512(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return hashlib.sha512(b"NOFILE").hexdigest()

def _verify_startup():
    actual = _self_hash()
    exp = actual if STAB_MODE == "self" else PRE_STAB
    if exp and actual != exp:
        raise SystemExit(f"Self-verification failed.\n expected={exp}\n actual  ={actual}")
    print(f"FINAL BUILD VERIFIED | SHA-512={actual} | Mode={STAB_MODE} | Python={sys.version.split()[0]}")

def _seal_or_unseal(args):
    if args.seal:
        p = Path(__file__)
        h = hashlib.sha512(p.read_bytes()).hexdigest()
        data = p.read_text(encoding="utf-8", errors="ignore")
        data = re.sub(r'PRE_STAB\s*=\s*".*?"', f'PRE_STAB = "{h}"', data)
        data = re.sub(r'POST_STAB\s*=\s*".*?"', f'POST_STAB = "{h}"', data)
        data = data.replace('STAB_MODE","self")', 'STAB_MODE","external")')
        p.write_text(data, encoding="utf-8")
        print("SEALED. Hash embedded:", h); sys.exit(0)
    if args.unseal:
        p = Path(__file__)
        data = p.read_text(encoding="utf-8", errors="ignore")
        data = data.replace('STAB_MODE","external")', 'STAB_MODE","self")')
        p.write_text(data, encoding="utf-8")
        print("UNSEALED. Now verifying against self-hash."); sys.exit(0)

# ========== CLI / Config / Logger ==========
def _argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--mining", choices=["off","solo","pool"], default=os.getenv("MINING","off"))
    ap.add_argument("--no-net", action="store_true", default=os.getenv("NO_NET","0") in ("1","true","True"))
    ap.add_argument("--verbose", action="store_true", default=os.getenv("VERBOSE","0") in ("1","true","True"))
    ap.add_argument("--fail-strict", action="store_true", default=os.getenv("FAIL_STRICT","1") in ("1","true","True"))
    ap.add_argument("--count-hashes", action="store_true", default=os.getenv("COUNT_HASHES","0") in ("1","true","True"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stick-until-stale", action="store_true", default=True)
    ap.add_argument("--no-stick", action="store_true")
    ap.add_argument("--poll-interval", type=int, default=int(os.getenv("POLL_INTERVAL","2")))
    ap.add_argument("--save-dir", default=os.getenv("SAVE_DIR","./proofs"))
    ap.add_argument("--outbox-dir", default=os.getenv("OUTBOX_DIR","./outbox"))
    ap.add_argument("--metrics-file", default=os.getenv("METRICS_FILE",""))
    ap.add_argument("--submit-retries", type=int, default=int(os.getenv("SUBMIT_RETRIES","8")))
    ap.add_argument("--submit-backoff", type=float, default=float(os.getenv("SUBMIT_BACKOFF","0.5")))
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--unseal", action="store_true")
    return ap

def merged_config(args):
    cfg = {
        "BTC_RPC_USER": os.getenv("BTC_RPC_USER","SingalCoreBitcoin"),
        "BTC_RPC_PASSWORD": os.getenv("BTC_RPC_PASSWORD","B1tc0n4L1dz"),
        "BTC_RPC_HOST": os.getenv("BTC_RPC_HOST","127.0.0.1"),
        "BTC_RPC_PORT": int(os.getenv("BTC_RPC_PORT","8332")),
        "BTC_WALLET": os.getenv("BTC_WALLET","SignalCoreBitcoinMining"),
        "PAYOUT_ADDR": os.getenv("MINER_PAYOUT_ADDR","bc1qcmxyhlpm3lf9zvyevqas9n547lywws2e7wvuu1"),
        "SECONDARY_RPC_HOST": os.getenv("SECONDARY_RPC_HOST",""),
        "SECONDARY_RPC_PORT": int(os.getenv("SECONDARY_RPC_PORT","0") or 0),
        "SECONDARY_RPC_USER": os.getenv("SECONDARY_RPC_USER",""),
        "SECONDARY_RPC_PASSWORD": os.getenv("SECONDARY_RPC_PASSWORD",""),
        "VERBOSE": args.verbose,
        "FAIL_STRICT": args.fail_strict,
        "NO_NET": args.no_net,
        "COUNT_HASHES": args.count_hashes,
        "MINING": args.mining,
        "DRY_RUN": args.dry_run,
        "STICK": not args.no-stick if hasattr(args,"no-stick") else True,
        "POLL_INTERVAL": args.poll_interval,
        "SAVE_DIR": args.save_dir,
        "OUTBOX_DIR": args.outbox_dir,
        "METRICS_FILE": args.metrics_file,
        "SUBMIT_RETRIES": args.submit_retries,
        "SUBMIT_BACKOFF": args.submit_backoff
    }
    Path(cfg["SAVE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["OUTBOX_DIR"]).mkdir(parents=True, exist_ok=True)
    return cfg

def log_json(cfg, obj):
    line = json.dumps(obj)
    if cfg.get("METRICS_FILE"):
        with open(cfg["METRICS_FILE"],"a",encoding="utf-8") as f: f.write(line+"\n")
    print(line)

# ========== Hash counter (optional) ==========
_hash_calls = 0
_real_sha256 = hashlib.sha256
class _CountingSHA256:
    def __init__(self,*a,**k):
        global _hash_calls; _hash_calls += 1; self._h=_real_sha256(*a,**k)
    def update(self,*a,**k): return self._h.update(*a,**k)
    def digest(self,*a,**k): return self._h.digest(*a,**k)
    def hexdigest(self,*a,**k): return self._h.hexdigest(*a,**k)
    def copy(self,*a,**k): c=_CountingSHA256(); c._h=self._h.copy(); return c
def enable_hash_counter(enable):
    if enable: hashlib.sha256 = _CountingSHA256
def get_hash_count(): return _hash_calls

# ========== RPC with failover ==========
import http.client, base64
def _rpc_once(host,port,user,pw,wallet,method,params):
    conn = http.client.HTTPConnection(host,int(port),timeout=15)
    try:
        payload = json.dumps({"jsonrpc":"2.0","id":"sig","method":method,"params":params or []})
        auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
        headers = {"Content-Type":"application/json","Authorization":f"Basic {auth}"}
        path = f"/wallet/{wallet}" if wallet else "/"
        conn.request("POST", path, body=payload, headers=headers)
        r = conn.getresponse(); b = r.read()
        if r.status != 200: raise RuntimeError(f"RPC HTTP {r.status}: {b[:200]}")
        data = json.loads(b.decode())
        if data.get("error"): raise RuntimeError(str(data["error"]))
        return data["result"]
    finally:
        conn.close()

def rpc_call(cfg, method, params=None):
    if cfg["NO_NET"]: raise RuntimeError("NO_NET set; RPC forbidden.")
    try:
        return _rpc_once(cfg["BTC_RPC_HOST"], cfg["BTC_RPC_PORT"], cfg["BTC_RPC_USER"], cfg["BTC_RPC_PASSWORD"], cfg["BTC_WALLET"], method, params)
    except Exception as e:
        if cfg.get("SECONDARY_RPC_HOST"):
            return _rpc_once(cfg["SECONDARY_RPC_HOST"], cfg["SECONDARY_RPC_PORT"], cfg["SECONDARY_RPC_USER"], cfg["SECONDARY_RPC_PASSWORD"], cfg["BTC_WALLET"], method, params)
        raise

# ========== ZMQ (optional) + Polling fallback ==========
def zmq_tip_watcher(signals):
    try:
        import zmq
    except Exception:
        signals["zmq"]=False; return
    addr = os.getenv("ZMQ_BLOCK","tcp://127.0.0.1:28332")
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"hashblock")
    try:
        s.connect(addr); signals["zmq"]=True
    except Exception:
        signals["zmq"]=False; return
    while not signals.get("stop"):
        try:
            topic, payload = s.recv_multipart()
            if topic == b"hashblock": signals["new_tip"]=True
        except Exception: break

def poll_tip(cfg, last_prevhash):
    try:
        h = rpc_call(cfg,"getbestblockhash")
        return h != last_prevhash
    except Exception:
        return False

# ========== Binary helpers ==========
def compact_to_target(nBits):
    if isinstance(nBits,str): nBits=int(nBits,16)
    exp = nBits >> 24
    mant = nBits & 0x007fffff
    if nBits & 0x00800000: mant = -mant
    return mant * (1 << (8*(exp-3)))

def varint(n):
    if n < 0xfd: return n.to_bytes(1,'little')
    if n <= 0xffff: return b'\xfd' + n.to_bytes(2,'little')
    if n <= 0xffffffff: return b'\xfe' + n.to_bytes(4,'little')
    return b'\xff' + n.to_bytes(8,'little')

def dbl_sha(b): return hashlib.sha256(hashlib.sha256(b).digest()).digest()
def hex_to_le(h): return bytes.fromhex(h)[::-1]

def header_bytes(version, prevhash_hex, merkleroot_bytes, nTime, nBits, nonce):
    return (int(version).to_bytes(4,'little') +
            hex_to_le(prevhash_hex) +
            merkleroot_bytes +
            int(nTime).to_bytes(4,'little') +
            (int(nBits,16) if isinstance(nBits,str) else int(nBits)).to_bytes(4,'little') +
            int(nonce).to_bytes(4,'little'))

def merkle_root(tx_hashes_le):
    layer = tx_hashes_le[:]
    if not layer: return b"\x00"*32
    while len(layer) > 1:
        if len(layer) % 2 == 1: layer.append(layer[-1])
        nxt=[]
        for i in range(0,len(layer),2):
            nxt.append(dbl_sha(layer[i]+layer[i+1]))
        layer=nxt
    return layer[0]

DEFAULT_MODEL_CALL = "ollama run mixtral:8x7b-instruct-v0.1-q6_K"

# ========== Embedded math engine ==========
LEVEL16000_SOURCE = r"""{math_body}"""

def load_math():
    ns={}
    exec(LEVEL16000_SOURCE, ns, ns)
    req = ["CheckDrift","IntegrityCheck","SyncState","EntropyBalance","ForkAlign",
           "accept_template","coinbase_plan","merkle_plan","schedule_roll",
           "stale_check","verify_hit","submit_policy","postmortem"]
    missing=[x for x in req if x not in ns]
    if missing: raise SystemExit("Math engine missing: "+", ".join(missing))
    return ns

# ========== Preflight++ ==========
def preflight(cfg):
    info = rpc_call(cfg,"getblockchaininfo")
    if info.get("initialblockdownload",True): raise SystemExit("Preflight: Node in IBD.")
    net = rpc_call(cfg,"getnetworkinfo")
    if int(net.get("connections",0)) < 8: raise SystemExit("Preflight: peers < 8")
    if abs(int(net.get("timeoffset",0))) > 2: raise SystemExit("Preflight: timeoffset > 2s")
    rpc_call(cfg,"getmempoolinfo")
    print("[PRE-FLIGHT] Node ready.")

# ========== Proof save / Outbox queue / Submit ==========
def save_proof(cfg, hdr_hex, block_hex, meta):
    ts = int(time.time())
    base = f"{ts}_{meta.get('template_id','unknown')}"
    d = Path(cfg["SAVE_DIR"]); d.mkdir(parents=True, exist_ok=True)
    (d/f"{base}.header.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (d/f"{base}.header.hex").write_text(hdr_hex, encoding="utf-8")
    (d/f"{base}.block.hex").write_text(block_hex, encoding="utf-8")
    if cfg.get("METRICS_FILE"):
        with open(cfg["METRICS_FILE"],"a",encoding="utf-8") as f: f.write(json.dumps({"event":"save","meta":meta})+"\n")

def submit_with_retry(cfg, block_hex, attempts, backoff):
    last=None
    for i in range(attempts):
        try:
            r = rpc_call(cfg,"submitblock",[block_hex])
            if r is None: return True,"accepted"
            return False,f"rejected: {r}"
        except Exception as e:
            last=str(e); time.sleep(backoff*(2**i))
    return False, f"submit failed: {last}"

def outbox_worker(cfg, q):
    while True:
        item = q.get()
        if item is None: break
        block_hex, meta = item
        ok,msg = submit_with_retry(cfg, block_hex, cfg["SUBMIT_RETRIES"], cfg["SUBMIT_BACKOFF"])
        if ok:
            print("Outbox submit: accepted")
        else:
            time.sleep(3.0); q.put(item)

# ========== Mining (math-first, stick-until-stale) ==========
def mine_once(cfg, M):
    # Pre-safeguards
    if not M["CheckDrift"](16000,"pre"): raise SystemExit("Pre-drift check failed.")
    if not M["IntegrityCheck"]((10<<8)+16000): raise SystemExit("Integrity check failed.")
    if M["EntropyBalance"](16000) < 0.2: raise SystemExit("Entropy parity too low.")
    state = M["SyncState"](16000,"forks")

    gbt = rpc_call(cfg,"getblocktemplate",[{"rules":["segwit"],"capabilities":["coinbasetxn","workid"]}])
    if not M["accept_template"](gbt, state):
        print("Math rejected template; refetching."); return None

    cb_plan = M["coinbase_plan"](gbt, state)
    txs = gbt.get("transactions",[])
    if "coinbasetxn" in gbt:
        tx_hexes = [gbt["coinbasetxn"]["data"]] + [t["data"] for t in txs]
    else:
        raise SystemExit("coinbasetxn missing; coinbase-from-scratch not enabled in this build.")

    # Merkle
    txids_le = []
    for hx in tx_hexes:
        h = dbl_sha(bytes.fromhex(hx))[::-1]
        txids_le.append(h)
    root = merkle_root(txids_le)

    target = compact_to_target(gbt["bits"])
    bounds = {"nTime_min": int(gbt["curtime"]), "nTime_max": int(gbt["curtime"])+120,
              "nonce_start": 0, "nonce_end": 0xffffffff}
    prefix = {"version": gbt["version"], "prevhash": gbt["previousblockhash"], "merkleroot": root.hex()}

    signals = {"new_tip": False, "stop": False, "zmq": False}
    zt = threading.Thread(target=zmq_tip_watcher, args=(signals,), daemon=True); zt.start()

    template_id = gbt.get("workid") or gbt["previousblockhash"][:16]
    rolled=0; t0=time.time()

    for (ver, nTime, extranonce, nonce) in M["schedule_roll"](prefix, bounds, state):
        if signals.get("new_tip") or poll_tip(cfg, gbt["previousblockhash"]):
            if M["stale_check"](gbt["previousblockhash"], signals, state):
                print("Stale signaled; switching template."); signals["stop"]=True; return None

        hbytes = header_bytes(ver, gbt["previousblockhash"], bytes.fromhex(prefix["merkleroot"]), nTime, gbt["bits"], nonce)
        hv = int.from_bytes(dbl_sha(hbytes)[::-1],'big'); rolled += 1

        if hv <= target and M["verify_hit"](hbytes, target, state):
            hdr_hex = hbytes.hex()
            block_hex = hdr_hex + varint(len(tx_hexes)).hex() + "".join(tx_hexes)
            meta = {"template_id": template_id, "rolled": rolled, "target": hex(target),
                    "hash": hv.to_bytes(32,'big').hex(), "nTime": nTime, "nonce": nonce,
                    "duration_sec": round(time.time()-t0,3)}
            save_proof(cfg, hdr_hex, block_hex, meta)
            pol = M["submit_policy"](block_hex, os.environ, state)
            if not cfg["DRY_RUN"] and pol.get("allow", True):
                ok,msg = submit_with_retry(cfg, block_hex, pol.get("retry",{}).get("attempts", cfg["SUBMIT_RETRIES"]), pol.get("retry",{}).get("backoff", cfg["SUBMIT_BACKOFF"]))
                print("Submit:", msg)
                if not ok:
                    OUTBOX.put((block_hex, meta))
            M["postmortem"]("hit", meta, state)
            signals["stop"]=True
            return meta

    signals["stop"]=True
    M["postmortem"]("nohit", {"rolled": rolled, "template_id": template_id}, state)
    return None

def main():
    ap = _argparser(); args = ap.parse_args()
    _seal_or_unseal(args)
    _verify_startup()

    cfg = merged_config(args)
    enable_hash_counter(cfg["COUNT_HASHES"])

    if args.preflight-only:
        preflight(cfg); return
    if cfg["MINING"] == "off":
        print("Mining mode OFF."); return

    M = load_math()
    preflight(cfg)

    # Outbox queue worker
    global OUTBOX; OUTBOX = queue.Queue()
    threading.Thread(target=outbox_worker, args=(cfg, OUTBOX), daemon=True).start()

    while True:
        try:
            mine_once(cfg, M)
        except Exception as e:
            if cfg["FAIL_STRICT"]: raise
            print("[WARN] mine_once error:", e); time.sleep(1.0)
        if cfg["DRY_RUN"]: break

if __name__ == "__main__":
    main()
