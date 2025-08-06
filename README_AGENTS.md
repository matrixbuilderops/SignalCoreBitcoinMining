# 🧠 Project: SignalCoreBitcoinMining — Agent Task Instructions

## 📦 FILE STRUCTURE OVERVIEW

| File Name                  | Purpose                                                 |
|---------------------------|---------------------------------------------------------|
| `Bitcoin Core Node RPC.txt` | Contains wallet info, ZMQ ports, and credentials       |
| `Model call.txt`            | Shell command to invoke the local Ollama model         |
| `math.txt`                  | Core math for validation and recursion logic           |
| `ai_interface.py`           | Script that interfaces AI with mining engine           |
| `block_listener.py`         | Subscribes to ZMQ events (block, tx)                   |
| `math_module.py`            | Holds Knuth logic, stabilizers, entropy checks         |
| `mining_controller.py`      | Orchestrates mining ops, level control, validation     |
| `orchestrator.py`           | High-level controller, AI interfacing, auto-scaling    |

## 🎯 OBJECTIVE

Use all files and math provided to **build a fully autonomous Bitcoin mining engine**, orchestrated via local AI model, following this logic:

1. **Block Listener** captures ZMQ events instantly
2. **AI Interface** uses `Ollama` to call the model with math/commands
3. **Math Module** enforces level parameters (e.g., Knuth(10,3,x))
4. **Mining Controller** instructs wallet to mine and monitors results
5. **Orchestrator** supervises the full flow, retrying or scaling as needed

## ✅ CRITICAL REQUIREMENTS

- No hallucinations — AI should be fed real-time data via scripts
- AI should know block boundaries and only compute after each ends
- Scripts should intercommunicate with minimal I/O overhead
- Include a toggle to suppress most terminal output once validated
- One script must validate logic and upload successfully to the blockchain

## 🧪 VALIDATION STACK (Run in CI)

These packages are included locally and must be used for validation:

- `black`
- `flake8`
- `pylint`
- `mypy`
- `bandit`
- `z3`
- `hypothesis`
- `coverage.py`
- `vulture`
- `interrogate`

## 🛠️ TASKS TO PERFORM

- Fill in all placeholder logic using `math.txt` as the driver
- AI should know its role and communicate with `block_listener.py`
- Final output must pass **100% of the validators above**

## 🔗 STARTUP

- Model: `ollama run mixtral:8x7b-instruct-v0.1-q6_K`
- RPC Info: Use credentials in `Bitcoin Core Node RPC.txt`

