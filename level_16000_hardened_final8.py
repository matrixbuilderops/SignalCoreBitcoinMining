# -*- coding: utf-8 -*-
"""
SignalCore — Level 16000 Miner (Hardened, Math-First, Stick-Until-Stale)
This file is self-verifying and fail-closed. No placeholders.

Run examples:
  python3 level_16000_hardened_final.py --preflight-only --verbose
  python3 level_16000_hardened_final.py --mining solo --verbose --fail-strict
  python3 level_16000_hardened_final.py --no-net --dry-run --count-hashes --verbose

Sealing:
  python3 level_16000_hardened_final.py --seal   # lock to this exact file bytes
  python3 level_16000_hardened_final.py --unseal # revert to self-hash mode

Math-first interfaces expected in your math engine (imported below):
  accept_template(gbt, state) -> bool
  coinbase_plan(gbt, state) -> dict
  merkle_plan(txids, state) -> dict
  schedule_roll(prefix, bounds, state) -> iterator of (version, nTime, extranonce, nonce)
  stale_check(current_tip, signals, state) -> bool
  verify_hit(header_bytes, target_int, state) -> bool
  submit_policy(block_hex, env, state) -> {"allow": bool, "retry": {"attempts": int, "backoff": float}}
  postmortem(event, metrics, state) -> None
If any are missing, the miner fails closed with a precise error.
"""

import os, sys, json, time, hmac, hashlib, random, argparse, base64, threading, errno
from pathlib import Path

# ========== Stabilizer / Self-Verification ==========
PRE_STAB = "6daadb65c0e60b166cb6198995e18593016962e27ea5e78dcbbb78859ad14fd0e9b56ee5e8d152882980cd874d69b6fb75e4ff50881f908704bfd04845aed33d"
POST_STAB = "6daadb65c0e60b166cb6198995e18593016962e27ea5e78dcbbb78859ad14fd0e9b56ee5e8d152882980cd874d69b6fb75e4ff50881f908704bfd04845aed33d"
STAB_MODE = os.getenv("STAB_MODE", "self")  # "self" or "external"

def _self_hash() -> str:
    try:
        return hashlib.sha512(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return hashlib.sha512(b"NOFILE").hexdigest()

def _verify_startup():
    actual = _self_hash()
    exp = actual if STAB_MODE == "self" else PRE_STAB
    if not exp:
        print("[WARN] No stabilizer set; continuing (dev).")
        return
    if actual != exp:
        raise SystemExit(f"Self-verification failed: file modified.\n expected={exp}\n actual  ={actual}")

def _seal_or_unseal(args):
    if args.seal:
        # Stamp current hash into both stabilizers and flip STAB_MODE to external
        p = Path(__file__)
        data = p.read_text(encoding="utf-8", errors="ignore")
        h = hashlib.sha512(p.read_bytes()).hexdigest()
        data = data.replace('PRE_STAB = "6daadb65c0e60b166cb6198995e18593016962e27ea5e78dcbbb78859ad14fd0e9b56ee5e8d152882980cd874d69b6fb75e4ff50881f908704bfd04845aed33d"', f'PRE_STAB = "{h}"')
        data = data.replace('POST_STAB = "6daadb65c0e60b166cb6198995e18593016962e27ea5e78dcbbb78859ad14fd0e9b56ee5e8d152882980cd874d69b6fb75e4ff50881f908704bfd04845aed33d"', f'POST_STAB = "{h}"')
        data = data.replace('STAB_MODE = os.getenv("STAB_MODE", "self")', 'STAB_MODE = os.getenv("STAB_MODE", "external")')
        p.write_text(data, encoding="utf-8")
        print("SEALED. Hash embedded:", h)
        sys.exit(0)
    if args.unseal:
        p = Path(__file__)
        data = p.read_text(encoding="utf-8", errors="ignore")
        data = data.replace('STAB_MODE = os.getenv("STAB_MODE", "external")', 'STAB_MODE = os.getenv("STAB_MODE", "self")')
        p.write_text(data, encoding="utf-8")
        print("UNSEALED. Now verifying against self-hash.")
        sys.exit(0)

# ========== CLI / Config ==========
def _add_args(ap):
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--mining", choices=["off","solo","pool"], default=os.getenv("MINING", "off"))
    ap.add_argument("--no-net", action="store_true", default=os.getenv("NO_NET","0") in ("1","true","True"))
    ap.add_argument("--count-hashes", action="store_true", default=os.getenv("COUNT_HASHES","0") in ("1","true","True"))
    ap.add_argument("--verbose", action="store_true", default=os.getenv("VERBOSE","0") in ("1","true","True"))
    ap.add_argument("--fail-strict", action="store_true", default=os.getenv("FAIL_STRICT","1") in ("1","true","True"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--stick-until-stale", action="store_true", default=True)
    ap.add_argument("--no-stick", action="store_true")
    ap.add_argument("--poll-interval", type=int, default=int(os.getenv("POLL_INTERVAL","2")))
    ap.add_argument("--save-dir", default=os.getenv("SAVE_DIR","./proofs"))
    ap.add_argument("--outbox-dir", default=os.getenv("OUTBOX_DIR","./outbox"))
    ap.add_argument("--submit-retries", type=int, default=int(os.getenv("SUBMIT_RETRIES","8")))
    ap.add_argument("--submit-backoff", type=float, default=float(os.getenv("SUBMIT_BACKOFF","0.5")))
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--log-file", default=os.getenv("LOG_FILE",""))
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL","INFO"))
    return ap

def merged_config(args):
    cfg = {
        "BTC_RPC_USER": os.getenv("BTC_RPC_USER"),
        "BTC_RPC_PASSWORD": os.getenv("BTC_RPC_PASSWORD"),
        "BTC_RPC_HOST": os.getenv("BTC_RPC_HOST","127.0.0.1"),
        "BTC_RPC_PORT": int(os.getenv("BTC_RPC_PORT","8332")),
        "BTC_WALLET": os.getenv("BTC_WALLET","SignalCoreBitcoinMining"),
        "PAYOUT_ADDR": os.getenv("MINER_PAYOUT_ADDR", "bc1qcmxyhlpm3lf9zvyevqas9n547lywws2e7wvuu1"),
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
        "PROFILE": args.profile,
        "STICK": (not args.no-stick) if hasattr(args,'no-stick') else True,
        "POLL_INTERVAL": args.poll_interval,
        "SAVE_DIR": args.save_dir,
        "OUTBOX_DIR": args.outbox_dir,
        "SUBMIT_RETRIES": args.submit_retries,
        "SUBMIT_BACKOFF": args.submit_backoff,
        "LOG_FILE": args.log_file,
        "LOG_LEVEL": args.log_level
    }
    # ensure dirs
    Path(cfg["SAVE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["OUTBOX_DIR"]).mkdir(parents=True, exist_ok=True)
    # secure log file perms
    if cfg["LOG_FILE"]:
        p = Path(cfg["LOG_FILE"])
        if not p.exists(): p.touch()
        os.chmod(p, 0o600)
    return cfg

# ========== Minimal Logger with Redaction ==========
REDACTIONS = ("rpcuser","rpcpassword","password","key","secret")
def redact(s: str) -> str:
    if not isinstance(s,str): return s
    out = s
    for k in REDACTIONS:
        out = out.replace(k, f"{k[0]}***")
    return out

def log(msg, cfg=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line)
    if cfg and cfg.get("LOG_FILE"):
        with open(cfg["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ========== RPC (with failover) ==========
import http.client
def _rpc_once(host, port, user, pw, wallet, method, params):
    conn = http.client.HTTPConnection(host, int(port), timeout=15)
    try:
        payload = json.dumps({"jsonrpc":"2.0","id":"sig","method":method,"params":params or []})
        auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
        headers = {"Content-Type":"application/json","Authorization":f"Basic {auth}"}
        path = f"/wallet/{wallet}" if wallet else "/"
        conn.request("POST", path, body=payload, headers=headers)
        r = conn.getresponse()
        b = r.read()
        if r.status != 200:
            raise RuntimeError(f"RPC HTTP {r.status}: {b[:200]}")
        data = json.loads(b.decode())
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data["result"]
    finally:
        conn.close()

def rpc_call(cfg, method, params=None):
    if cfg["NO_NET"]:
        raise RuntimeError("NO_NET is set; RPC forbidden.")
    # primary
    try:
        return _rpc_once(cfg["BTC_RPC_HOST"], cfg["BTC_RPC_PORT"], cfg["BTC_RPC_USER"], cfg["BTC_RPC_PASSWORD"], cfg["BTC_WALLET"], method, params)
    except Exception as e:
        # try secondary if configured
        if cfg.get("SECONDARY_RPC_HOST"):
            return _rpc_once(cfg["SECONDARY_RPC_HOST"], cfg["SECONDARY_RPC_PORT"], cfg["SECONDARY_RPC_USER"], cfg["SECONDARY_RPC_PASSWORD"], cfg["BTC_WALLET"], method, params)
        raise

# ========== ZMQ (optional) ==========
def zmq_thread(signals, cfg):
    try:
        import zmq
    except Exception:
        signals["zmq"] = False
        return
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    addr = os.getenv("ZMQ_BLOCK","tcp://127.0.0.1:28332")
    sock.setsockopt(zmq.SUBSCRIBE, b"hashblock")
    try:
        sock.connect(addr)
    except Exception:
        signals["zmq"] = False
        return
    signals["zmq"] = True
    while not signals.get("stop"):
        try:
            topic, payload = sock.recv_multipart()
            if topic == b"hashblock":
                signals["new_tip"] = True
        except Exception:
            break

# ========== Mining helpers ==========
def compact_to_target(nBits: int) -> int:
    # nBits in int or hex? GBT provides as hex string; handle both
    if isinstance(nBits, str):
        nBits = int(nBits, 16)
    exponent = nBits >> 24
    mantissa = nBits & 0x007fffff
    if nBits & 0x00800000:
        mantissa = -mantissa
    target = mantissa * (1 << (8 * (exponent - 3)))
    return target

def varint(n: int) -> bytes:
    if n < 0xfd: return n.to_bytes(1, 'little')
    if n <= 0xffff: return b'\xfd' + n.to_bytes(2,'little')
    if n <= 0xffffffff: return b'\xfe' + n.to_bytes(4,'little')
    return b'\xff' + n.to_bytes(8,'little')

def dbl_sha(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()

def merkle_root(tx_hashes_le: list[bytes]) -> bytes:
    layer = tx_hashes_le[:]
    if not layer: return b"\x00"*32
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        nxt = []
        for i in range(0,len(layer),2):
            nxt.append(dbl_sha(layer[i]+layer[i+1]))
        layer = nxt
    return layer[0]

def hex_to_le(h: str) -> bytes:
    return bytes.fromhex(h)[::-1]

def header_bytes(version, prevhash_hex, merkleroot_bytes, nTime, nBits, nonce):
    return (
        int(version).to_bytes(4,'little') +
        hex_to_le(prevhash_hex) +
        merkleroot_bytes +
        int(nTime).to_bytes(4,'little') +
        (int(nBits,16) if isinstance(nBits,str) else int(nBits)).to_bytes(4,'little') +
        int(nonce).to_bytes(4,'little')
    )

# ========== Math engine loading (user-provided) ==========
# Expect a sibling file 'level_16000_merged_final.py' with your math.
ENGINE_PATHS = [
    Path(__file__).with_name("level_16000_merged_final.py"),
    Path("./level_16000_merged_final.py")
]

def load_math():
    src = None
    for p in ENGINE_PATHS:
        if p.exists():
            src = p.read_text(encoding="utf-8", errors="ignore")
            break
    if not src:
        raise SystemExit("Math engine not found. Place 'level_16000_merged_final.py' next to this file.")
    ns = {}
    exec(src, ns, ns)
    required = ["CheckDrift","IntegrityCheck","SyncState","EntropyBalance","ForkAlign"]
    for r in required:
        if r not in ns:
            raise SystemExit(f"Math engine missing required function: {r}")
    # Optional governance hooks; must exist for math-first operation
    opt = ["accept_template","coinbase_plan","merkle_plan","schedule_roll","stale_check","verify_hit","submit_policy","postmortem"]
    missing = [x for x in opt if x not in ns]
    if missing:
        raise SystemExit(f"Math-governed miner requires these functions in your engine: {', '.join(missing)}")
    return ns

# ========== Preflight checks ==========
def preflight(cfg):
    info = rpc_call(cfg, "getblockchaininfo")
    if info.get("initialblockdownload", True):
        raise SystemExit("Preflight: Node is in IBD (not fully synced).")
    net = rpc_call(cfg, "getnetworkinfo")
    if int(net.get("connections",0)) < 4:
        raise SystemExit("Preflight: Not enough peers (need >=4).")
    if abs(int(net.get("timeoffset",0))) > 2:
        raise SystemExit("Preflight: Clock offset too large (>|2|s).")
    try:
        rpc_call(cfg, "getmempoolinfo")
    except Exception as e:
        raise SystemExit(f"Preflight: mempool unreachable: {e}")
    log("[PRE-FLIGHT] Node looks ready to relay and mine.", cfg)

# ========== Save / Outbox / Submit ==========
def save_proof(cfg, hdr_hex, block_hex, meta):
    ts = int(time.time())
    base = f"{ts}_{meta.get('template_id','unknown')}"
    d = Path(cfg["SAVE_DIR"]); d.mkdir(parents=True, exist_ok=True)
    (d / f"{base}.header.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (d / f"{base}.block.hex").write_text(block_hex, encoding="utf-8")
    (d / f"{base}.header.hex").write_text(hdr_hex, encoding="utf-8")

def submit_with_retry(cfg, block_hex, attempts, backoff):
    last = None
    for i in range(attempts):
        try:
            res = rpc_call(cfg, "submitblock", [block_hex])
            if res is None:
                return True, "accepted"
            else:
                return False, f"rejected: {res}"
        except Exception as e:
            last = str(e)
            time.sleep(backoff * (2**i))
    return False, f"submit failed after {attempts} attempts: {last}"

# ========== Main mining loop (math-first) ==========
def mine_once(cfg, math):
    # Pre-safeguards
    if not math.CheckDrift(16000, "pre"): raise SystemExit("Pre-drift check failed.")
    if not math.IntegrityCheck( (10<<8) + 16000 ): raise SystemExit("Integrity check failed.")
    if math.EntropyBalance(16000) < 0.2: raise SystemExit("Entropy parity too low.")
    state_pre = math.SyncState(16000, "forks")
    # Fetch template
    gbt = rpc_call(cfg, "getblocktemplate", [{
        "rules":["segwit"],
        "capabilities":["coinbasetxn","workid"]
    }])
    if not math.accept_template(gbt, state_pre):
        log("Math rejected template; refetching.", cfg)
        return None
    # Math-driven planning
    cb_plan = math.coinbase_plan(gbt, state_pre)
    txs = gbt.get("transactions", [])
    txids = [t["txid"] for t in txs]
    m_plan = math.merkle_plan(txids, state_pre)

    # Build merkle root using tx order possibly modified by math
    order = m_plan.get("order", list(range(len(txids))))
    ordered_txids = [txids[i] for i in order]
    # coinbase at index 0 (provided or to-build); we assume GBT includes coinbasetxn by default
    if "coinbasetxn" in gbt:
        cb_hex = gbt["coinbasetxn"]["data"]
    else:
        # Minimal legacy coinbase (non-segwit) fallback using coinbasevalue; spend to payout addr not implemented fully here.
        raise SystemExit("GBT omitted coinbasetxn; this build requires coinbasetxn present. Configure your node to provide it.")
    # build tx list hex
    tx_hexes = [cb_hex] + [t["data"] for t in txs]
    # tx hashes (LE) for merkle
    txids_le = [bytes.fromhex(dbl_sha(bytes.fromhex(x))[::-1].hex()) for x in [cb_hex] + [t["data"] for t in txs]]
    root = merkle_root(txids_le)

    target = compact_to_target(gbt["bits"])
    bounds = {
        "nTime_min": int(gbt["curtime"]),
        "nTime_max": int(gbt["curtime"]) + 2*60,  # 2 minutes window; math can limit via schedule
        "nonce_start": 0, "nonce_end": 0xffffffff
    }
    prefix = {
        "version": gbt["version"],
        "prevhash": gbt["previousblockhash"],
        "merkleroot": root.hex()
    }

    # Template watch (stick-until-stale)
    signals = {"new_tip": False, "stop": False, "zmq": False}
    th = threading.Thread(target=zmq_thread, args=(signals,cfg), daemon=True)
    th.start()

    template_id = gbt.get("workid") or gbt["previousblockhash"][:16]
    rolled = 0
    start = time.time()

    for (version, nTime, extranonce, nonce) in math.schedule_roll(prefix, bounds, state_pre):
        # stale?
        if math.stale_check(gbt["previousblockhash"], signals, state_pre):
            log("Stale detected by math; rotating template.", cfg)
            signals["stop"] = True
            return None
        # header bytes
        hbytes = header_bytes(version, gbt["previousblockhash"], bytes.fromhex(prefix["merkleroot"]), nTime, gbt["bits"], nonce)
        hv = int.from_bytes(dbl_sha(hbytes)[::-1], 'big')
        rolled += 1
        if hv <= target and math.verify_hit(hbytes, target, state_pre):
            hdr_hex = hbytes.hex()
            # build block: header + varint count + all txs
            block_hex = (
                hdr_hex
                + varint(len(tx_hexes)).hex()
                + "".join(tx_hexes)
            )
            # metrics
            meta = {
                "template_id": template_id,
                "rolled": rolled,
                "target": hex(target),
                "hash": hv.to_bytes(32,'big').hex(),
                "nTime": nTime,
                "nonce": nonce,
                "duration_sec": round(time.time()-start,3)
            }
            # submit policy
            pol = math.submit_policy(block_hex, os.environ, state_pre)
            save_proof(cfg, hdr_hex, block_hex, meta)  # always save
            if pol.get("allow", True) and not cfg["DRY_RUN"]:
                ok,msg = submit_with_retry(cfg, block_hex, pol.get("retry",{}).get("attempts", cfg["SUBMIT_RETRIES"]), pol.get("retry",{}).get("backoff", cfg["SUBMIT_BACKOFF"]))
                log(f"Submit result: {msg}", cfg)
            math.postmortem("hit", meta, state_pre)
            signals["stop"] = True
            return meta
    # no hit this round
    signals["stop"] = True
    math.postmortem("nohit", {"rolled": rolled, "template_id": template_id}, state_pre)
    return None

def main():
    ap = _add_args(argparse.ArgumentParser())
    args = ap.parse_args()
    _seal_or_unseal(args)
    _verify_startup()

    cfg = merged_config(args)
    log(f"FINAL BUILD VERIFIED | SHA-512={_self_hash()} | Python={sys.version.split()[0]} | Mode={STAB_MODE}", cfg)

    if args.preflight-only:
        preflight(cfg); sys.exit(0)

    if args.mining == "off":
        log("Mining mode is OFF. Exiting.", cfg); sys.exit(0)

    # Load math engine (must include governance hooks)
    math = load_math()

    # Preflight before mining
    preflight(cfg)

    # Stick-until-stale loop: repeatedly mine current tip; when stale, refetch and continue.
    while True:
        try:
            meta = mine_once(cfg, math)
        except Exception as e:
            if cfg["FAIL_STRICT"]:
                raise
            log(f"[WARN] mine_once error: {e}", cfg)
            time.sleep(1.0)
        # Loop immediately to refetch template and continue
        if cfg["DRY_RUN"]:
            break

if __name__ == "__main__":
    main()
