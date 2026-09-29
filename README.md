# CEREBRO ZERO

<p align="center">
  <img src="./cerebro_zero_final_white.png" width="120" alt="Cerebro Zero" />
</p>


## Autonomous Learning Cognitive Agent

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.4-green.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-770%2F770-brightgreen.svg)](tests/)
[![CI](https://github.com/lecodev-26/cerebro-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![No Pickle](https://img.shields.io/badge/Pickle-Zero-success.svg)](#serialization)
[![Made in Termux](https://img.shields.io/badge/Made%20in-Termux-orange.svg)](https://termux.dev/)

---

## 🎬 Demo

![Cerebro Zero in action](assets/demo.gif)

*A complete cognitive agent built from scratch on Termux (Android + NumPy). No frameworks. No shortcuts.*

---

## 📖 Description

**Cerebro Zero 5.0.0** is the current release: a **cognitive autonomous runtime** built on the verified V4 ML stack. It adds a deterministic cognitive cycle for context, memory, reasoning, planning, guarded actions, verification, learning and consolidation. No ML frameworks. No `pickle` in the active tree.

**Philosophy 5.0 — Cognitive Autonomy:**
> 2.1 = verifiable system.
> 3.0 = system that learns, plans and acts verifiably.
> 4.0 = system integrity: zero pickle, transactional learning, honest benchmarks.
> 5.0 = deterministic cognitive runtime: observe, reason, plan, guard, act, verify, learn and consolidate.

---

## 🧠 Cerebro Zero 5.0.0 — Cognitive Autonomous Runtime

V5 is the current release and extends the verified V4 foundation with an autonomous orchestration layer:

`observe → context → retrieve → reason → plan → guard → act → verify → learn → consolidate`

- **Cognitive cycle:** deterministic end-to-end runtime orchestration.
- **Memory:** working, episodic, semantic and procedural layers with consolidation.
- **Reasoning:** structured hypotheses and evidence with memory reuse.
- **Planning:** hierarchical plans with verification and plan repair.
- **Tools:** registry, schemas, permissions and risk-aware execution.
- **Context:** bounded context construction and compression.
- **Skills:** procedural skill registry, selection and usage tracking.
- **Evaluation:** outcome scoring, confidence error and calibration signals.
- **Learning:** experience replay, strategy tracking and reward accumulation.
- **Safety:** action/risk budgets and a safe simulated executor by default.
- **Benchmark:** reproducible long-running V5 benchmark for Android/Termux.

### V5 release qualification

- **Tests:** 770/770 passing in the Termux release environment.
- **Benchmark:** 1,000/1,000 successful cycles (100% success in the release benchmark).
- **Throughput:** 195.66 cycles/s in that 1,000-cycle run on the release environment.
- **Learning trace:** 1,000 episodic experiences and 1,000 skill uses recorded.
- **Package:** `5.0.0`, verified as installed.

V5 preserves the V4 ML/Transformer stack; historical V4 documentation remains available for release history and validation details.

---

## 🏆 V4 foundation achievements retained in V5

| Area | Status |
|------|--------|
| **Tests** | ✅ 770/770 passing |
| **Autograd** | ✅ Verified with gradient checking |
| **Transformer** | ✅ REAL backprop (17/17 gradients) |
| **Cognitive Pipeline** | ✅ CerebroV3 integrated (10 components) |
| **Planner 3.0** | ✅ Goal → Subgoals → Dependencies → Actions |
| **Learning Engine** | ✅ Transactional Commit/Rollback with weight restore |
| **RL basics** | ✅ Q-Learning + GridWorld (~99% success) |
| **Tokenizer 3.0** | ✅ Byte-level + BPE from scratch |
| **Tool System 3.0** | ✅ Contracts + permissions + schemas |
| **Reproducibility** | ✅ Environment fingerprints + config hash |
| **Serialization** | ✅ **Zero pickle** across active tree (JSON + NPY) |
| **Repository** | ✅ Clean tree with `legacy/` archive |
| **Sandbox** | ✅ Subprocess + timeout + temp cwd |
| **CI/CD** | ✅ Unified GitHub Actions CI across Python 3.10–3.14 |
| **CLI** | ✅ `cerebro-zero info/chat/plan/benchmark` |
| **Visual** | ✅ Rich terminal output + Pillow-generated plots |
| **E2E** | ✅ Real CerebroV3 benchmark + system tests |
| **Memory index** | ✅ Lexical pre-index + vectorized retrieval |
| **KV cache** | ✅ Incremental Transformer inference cache |

---

## 🎯 V4 foundation retained in V5

### **Mathematical core**
- Tensor with full autograd
- Correct backward broadcasting
- Exhaustive gradient checking
- Float32/float64 configurable

### **Transformer with REAL backprop**
- Multi-Head Attention with causal mask
- LayerNorm with correct gradients
- Residuals and FFN
- **Verified: 17/17 parameters with sampled gradient check**

### **Cognitive memories**
- **Working Memory**: limited capacity, TTL, priority, reinforcement
- **Procedural Memory**: learn *how* to do things
- **Semantic Memory**: similarity search
- **World Model**: persistent world state with nested paths

### **Planner 3.0**
- Decomposition: Goal → Subgoals → Dependencies → Actions
- Topological order (Kahn's algorithm)
- Cycle and broken-dependency detection
- Domain templates: train, evaluate, remember
- Step-by-step execution

### **Learning Engine (4.3 — transactional)**
- Records experiences: state, action, result, reward, error
- Experience replay
- **Snapshot / rollback** of weights
- **Golden rule:** the model is NEVER modified without objective improvement
- Commit on improvement, rollback on regression, rollback on error

### **RL basics**
- NxN GridWorld with obstacles
- Tabular Q-Learning with epsilon-greedy
- Training and no-exploration evaluation
- ~99% success rate after training

### **Tokenizer 3.0**
- **ByteTokenizer**: 256 bytes + 4 special tokens, handles any text
- **BPETokenizer**: Byte Pair Encoding from scratch
- Real compression (up to 6.5x on seen words)
- Handles emojis, accents, Chinese, rare symbols

### **Tool System 3.0**
- `ToolSpec`: declarative contract (input, output, permissions, capabilities)
- Schema validation (types, min/max, required, default)
- Search by capability and text
- Execution history and stats

### **Benchmark 4.0**
- Categories: math, memory, planning, rl, learning
- Scientific metrics: accuracy, exact_match, error_rate, latency_p50/p95
- Splits: KNOWN, UNSEEN, ADVERSARIAL
- Reproducible with seed

### **Reproducibility**
- `hash_estructura`: deterministic, order-independent
- `hash_archivo`: SHA256 chunked
- `EntornoCaptura`: Python, NumPy, platform, hardware, git commit
- `RegistroExperimento`: unique fingerprint per experiment
- Reproducibility verification with tolerance

### **Serialization (4.2 — zero pickle)**
- `core/serialization.py`: JSON + NPY for dicts with ndarrays
- `core/serialization_v2.py`: recursive support for nested dicts/lists
- Checksums SHA256, corruption detection
- Inspeccionable with `cat`
- **Zero `pickle` in the active tree** (legacy archive only)

### **Sandbox & security (4.8)**
- Subprocess isolation with temporary working directory
- Timeout enforcement and output limits
- POSIX CPU/file/process/file-descriptor limits where supported
- Python isolated mode and reduced child environment
- Persistent audit log
- Platform-aware: Termux/Android does not provide Linux namespaces/seccomp isolation through this Python layer

### **Repository cleanup (4.1)**
- All legacy code moved to `legacy/` with its own README
- Active tree: 20 clean modules
- Legacy tests separated in `tests/legacy/`

### **Visual layer**
- `utils/visual.py`: rich-based semantic helpers (ok/warn/error/info, tables, panels)
- `utils/plots.py`: Pillow-generated PNG learning curves
- Live progress bars during training

---

## 🚀 Installation

```bash
git clone https://github.com/lecodev-26/cerebro-zero.git
cd cerebro-zero
pip install -r requirements.txt
pytest tests/ -v
```

Requirements

```
numpy>=1.20
rich>=15.0
tqdm>=4.70
pillow>=12.0
```

Optional:

```
flask>=3.0        # for the legacy dashboard (moved to legacy/)
```

---

🖥️ CLI

```bash
# Version
python cli.py --version

# System status
python cli.py info

# One-shot chat
python cli.py chat "5 + 3"

# Interactive chat
python cli.py chat

# Generate a plan
python cli.py plan "Train the model"

# Run benchmark
python -m evaluation.benchmark_v5 --cycles 1000

# Run tests
python cli.py test
```

---

🧠 Usage example

```python
from v5.runtime import CerebroZeroV5

cerebro = CerebroZeroV5()

# Math (tool)
result = cerebro.run("5 + 3")
print(result)

# Cognitive cycle
result = cerebro.run("remember that the sky is blue")
print(result.cycle)

# Runtime stats
print(cerebro.stats())
```

---

📁 Structure

```
cerebro-zero/
├── agent/            # CerebroV3 (integration)
├── config/           # YAML config
├── core/             # Tensor, autograd, layers, transformer, serialization
├── datasets/         # Datasets
├── docs/             # Documentation
├── evaluation/       # V5 benchmark + reproducibility
├── v5/                # Cognitive autonomous runtime
├── language/         # Tokenizer 3.0, vocabulary
├── memory/           # Working, Procedural, World, Semantic
├── mobile/           # Optimizer, metrics, quantizer
├── models/           # Brain, MLP, Transformer
├── reasoning/        # Planner 3.0
├── security/         # Permissions, sandbox, process
├── tests/            # 770 active tests
├── tools/            # Tool System 3.0 + plugins
├── training/         # LM Trainer, Learning Engine, RL
├── utils/            # Visual (rich) + Plots (Pillow)
├── legacy/           # Historic code (archived, not executed)
├── assets/           # Demo GIF and images
├── cli.py            # CLI
└── pyproject.toml    # Packaging
```

---

📊 Benchmark V5

Run:

```bash
python -m evaluation.benchmark_v5 --cycles 1000
```

Metrics:

· Successful cognitive cycles
· Success rate
· Elapsed time and cycles/second
· Runtime statistics: episodic memory, learning experiences, skills and tools

---

🗺️ Roadmap

See ROADMAP.md for the full breakdown.

Completed

· ✅ Phases 1-20: Neural core (Tensor, autograd, layers)
· ✅ Phases 21-40: Central brain, Transformer, engineering
· ✅ Phases 2.1.1-2.1.8: Technical Integrity Release
· ✅ Phases 3.1-3.13: Autonomous Learning Cognitive Agent
· ✅ Phase 4.0.1: Honesty Fix (documentation audit)
· ✅ Phase 4.1: Repository Cleanup
· ✅ Phase 4.2: Serialization 2.0 (zero pickle)
· ✅ Phase 4.3: Learning Transactions (rollback)
· ✅ Visual Block F1-F2: rich + Pillow

Status

· Version: 5.0.0 — release complete
· Tests: see the live CI badge and `pytest --collect-only -q` for the current count
· V5 Cognitive Cycle: ✅
· V5 Multi-layer Memory: ✅
· V5 Reasoning + Evidence: ✅
· V5 Planning + Repair: ✅
· V5 Tool Permissions + Risk Budgets: ✅
· V5 Skills + Strategy Learning: ✅
· V5 Evaluation + Calibration: ✅
· V5 Replay + Experience Storage: ✅
· V5 Reproducible Benchmark: ✅
· V5 Android/Termux Qualification: ✅
· V5 Release Gate: ✅

---

⚠️ Known limitations

Cerebro Zero 5.0.0 is an educational laboratory with explicit technical limitations.

Benchmark

· The release benchmark now exercises the real CerebroV3 pipeline in `evaluation/benchmark_e2e.py`.
· The legacy handler benchmark remains as a component-level reference suite.

Security

· Subprocess, timeout, output, and POSIX resource controls are enabled where the platform supports them.
· Termux/Android does not expose Linux namespaces/seccomp isolation through this Python layer, so this is not a full OS sandbox.
· Network isolation is not claimed.

Gradient checking

· Transformer gradient checking remains sampled rather than exhaustive over every tensor element.

Performance

· Semantic memory now uses a lexical pre-index plus vectorized candidate scoring.
· Transformer inference has an incremental KV-cache.
· Other subsystems may still contain O(n) operations by design.

Legacy code

· `legacy/` contains historical code and old pickle-based artifacts.
· Legacy artifacts are excluded from the active package and active test tree.

---

🔎 Discoverability

Cerebro Zero is intentionally described with searchable terms used by the project itself: **from-scratch AI, cognitive agent, autonomous learning, Python, NumPy, autograd, Transformer, memory, planning, reinforcement learning, tokenizer, reproducibility, Termux, Android**. These are descriptive project keywords, not a claim about search ranking.

---

🤝 Contributing

See CONTRIBUTING.md.

---

📄 License

MIT — see LICENSE.

---

👨‍💻 Author

Manuel (lecodev-26)

· GitHub: @lecodev-26
· Email: axiomsystemsechepares@gmail.com

---

Built with 🧠 and ☕ from a Samsung A16 running Termux.
