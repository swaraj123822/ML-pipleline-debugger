---
title: ML Pipeline Debugger
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

<div align="center">

# 🔧 ML Pipeline Debugger

**An AI Agent Environment for Debugging Broken Deep Learning Pipelines**

[![Hackathon](https://img.shields.io/badge/Meta%20PyTorch%20×%20HuggingFace-Hackathon-orange?style=for-the-badge)](https://huggingface.co/spaces/subhrajit36/my-env)
[![Track](https://img.shields.io/badge/Track-OpenEnv-blue?style=for-the-badge)]()
[![Team](https://img.shields.io/badge/Team-DOT--DOT-purple?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.12+-green?style=for-the-badge&logo=python)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal?style=for-the-badge&logo=fastapi)]()

*An OpenEnv-compliant environment where an AI agent acts as an ML engineer — diagnosing and fixing broken training pipelines across three difficulty levels.*

</div>

---

## 🏗️ Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                        AI AGENT                             │
│         (OpenAI SDK / Custom Policy / RL Agent)             │
└───────────────────────────┬─────────────────────────────────┘
                            │  JSON Action Payload
                            │  e.g. {"action": "fix_reshape"}
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server  (port 7860)                    │
│                                                             │
│   ┌───────────────────────────────────────────────────────┐ │
│   │          Transport Layer (dual protocol)              │ │
│   │   WebSocket /ws   │  HTTP POST /reset, /step, /state  │ │
│   └─────────┬─────────┴──────────┬────────────────────────┘ │
│             │                    │                          |  
│   ┌─────────▼────────────────────▼────────────────────────┐ │
│   │         MLDebuggerEnvironment (OpenEnv)               │ │
│   │            reset() · step() · state()                 │ │
│   └─────────────────────┬─────────────────────────────────┘ │
│                         │                                   │
│   ┌─────────────────────▼─────────────────────────────────┐ │
│   │  ┌─────────────┐   ┌──────────────┐   ┌────────────┐  │ │
│   │  │  Simulator  │   │   Graders    │   │  Reward    │  │ │
│   │  │ (<50ms/step)│   │ (3 tasks)    │   │  Engine    │  │ │
│   │  └─────────────┘   └──────────────┘   └────────────┘  │ │
│   └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │  Observation (error_msg, metrics,
                            │  tensor_shapes, reward, done)
                            ▼
                         AI AGENT
```

No real model training occurs at any point. Every response — error messages, training logs, metrics — is authored in advance and stored as deterministic state machines. This design allows the environment to be **fully deterministic and GPU-free**.

---

## 🗂️ Project Structure

```
ml-pipeline-debugger/
├── server/
│   ├── app.py              # FastAPI server (HTTP + WebSocket /ws endpoints)
│   ├── environment.py      # Core OpenEnv MLDebuggerEnvironment logic
│   ├── simulator.py        # CPU-only simulation engine (<50ms per step)
│   ├── tasks.py            # 3 difficulty tasks with deterministic auto-graders
│   ├── reward.py           # Dense step-by-step reward function
│   └── requirements.txt    # Server-side dependencies
├── tests/
│   ├── test_env_loop.py    # End-to-end episode loop tests (reset/step/state)
│   ├── test_graders.py     # Deterministic grader correctness tests
│   └── test_simulator.py   # Simulation engine & crash detection tests
├── baseline/
│   ├── __init__.py         # Baseline agent package marker
│   └── requirements.txt    # Client-side dependencies (openai, websockets)
├── validation-scripts/
│   └── validate-submission.sh  # OpenEnv submission validation (ping + docker + openenv)
├── models.py               # Pydantic v2 Action/Observation/State contracts (shared)
├── inference.py            # Baseline LLM agent (OpenAI-compatible, multi-task)
├── client.py               # Reusable async/sync WebSocket client (MLDebuggerEnv)
├── openenv.yaml            # OpenEnv metadata & spec declaration
├── pyproject.toml          # Project config, dependencies, dev tools (ruff, pytest)
├── Dockerfile              # HF Spaces-ready Docker container config
└── README.md               
```

---

## 🎯 Tasks

### Task 1 — Easy: Tensor Shape Error

| Attribute | Detail |
|-----------|--------|
| **Problem** | A CNN's Conv2d layers output the wrong shape (512 vs 2304) for the subsequent Linear layer, crashing at forward pass with a dimension mismatch. |
| **Architecture** | `Conv2d(3,64,3,pad=0) → ReLU → MaxPool2d(5) → Flatten([512]) ← WRONG → Linear(512,10)` |
| **Correct action** | `fix_reshape` with `layer="flatten"`, `new_shape=[2304]` (64×6×6) |
| **Grader** | Evaluates correct layer identification, precise shape, and step efficiency |
| **Max steps** | 15 |
| **Success threshold** | ≥ 0.99 |

### Task 2 — Medium: Overfitting Problems

| Attribute | Detail |
|-----------|--------|
| **Problem** | A Transformer text classifier memorizes training data — train loss plummets but validation loss diverges and spikes by epoch 2. |
| **Architecture** | `Embedding(30522,256) → 4×TransformerEncoderLayer(d=256, 8 heads) → Linear(256,2)` — binary classifier with 95%/5% class imbalance |
| **Correct actions** | `add_augmentation` applying both `dropout` **and** `truncate_sequence` sequentially to close the val/train gap below 0.15 |
| **Grader** | Credits correctly started regularization strategies and heavily rewards closing the val/train loss gap safely |
| **Max steps** | 15 |
| **Success threshold** | ≥ 0.80 |

### Task 3 — Hard: Segmentation IoU 

| Attribute | Detail |
|-----------|--------|
| **Problem** | A SegNet architecture plateaus at IoU=0.52 due to a **frozen encoder**, unbalanced loss weights, and suboptimal learning rate. Target: **IoU > 0.85 within 5 steps**. |
| **Architecture** | `ResNet50-Encoder(FROZEN) → AdaptiveAttentionGate(512) → LightweightDecoder(256,128,64) → Sigmoid` — 5-class lung segmentation |
| **Dataset** | Cross-dataset: trained on **ChestX-ray14**, validated on **RSNA Pneumonia** |
| **Initial config** | `0.6×CrossEntropy + 0.4×Dice`, pre-trained encoder weights **locked** |
| **The Trap** | Without unfreezing the `ResNet50-Encoder`, IoU **cannot exceed ~0.72** regardless of hyperparameters. The agent must recognize that `toggle_layer_freeze` is essential — not a distractor. |
| **Correct actions** | `toggle_layer_freeze` (unfreeze `ResNet50-Encoder`) + `adjust_loss_weights` (Dice weight 0.55–0.75) + `tune_hyperparameters` (lr 0.0001–0.003) |
| **Grader** | Measures IoU achievement across tiers (0.55→0.65→0.75→0.85), rewards optimal lr and dice weight configuration, applies step-limit bonus |
| **Max steps** | 15 (but must reach target within **5 steps** for full credit) |
| **Success threshold** | ≥ 0.84 |

---

## 🔌 Action Space

A discriminated union of **6 typed actions**, validated by Pydantic v2:

| Action | Parameters | Task Relevance | Description |
|--------|-----------|----------------|-------------|
| `fix_reshape` | `layer: str, new_shape: list[int]` | **Easy** (primary) | Fix dimension mismatch in Linear/Flatten layers |
| `tune_hyperparameters` | `lr: float, batch_size: int, epochs: int` | **All tasks** | Adjust training hyperparameters (lr ∈ (0,1), batch_size = power-of-2, epochs ∈ [1,50]) |
| `add_augmentation` | `strategy: str` | **Medium** (primary) | Apply regularization (`dropout`, `weight_decay`, `truncate_sequence`, `horizontal_flip`, `mixup`) |
| `adjust_loss_weights` | `dice_weight: float, ce_weight: float` | **Hard** (required) | Rebalance Dice vs Cross-Entropy loss (must sum to 1.0) |
| `change_optimizer` | `optimizer: str, weight_decay: float` | Distractor | Switch optimizer (`Adam`, `SGD`, `RMSprop`) — valid but provides no improvement signal |
| `toggle_layer_freeze` | `layer_name: str, freeze: bool` | **Hard** (essential) | Freeze/unfreeze layer gradients — **critical** for unlocking hard task IoU ceiling |

> **Design Note:** `Toggle_layer_freeze` looks like a distractor but is actually essential. The `frozen ResNet50-Encoder` limits IoU to ~0.72, so tuning won’t help. `Agents` must notice the "FROZEN" hint and plateau to realize it needs `unfreezing`.

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task marker (`easy`, `medium`, `hard`) |
| `architecture_summary` | `str` | Text description of module composition, layers, and current state |
| `tensor_shapes` | `dict[str, list[int]]` | Expected vs actual input tensor dimension states |
| `error_trace` | `str \| null` | Python stacktrace simulation indicating crash events |
| `metrics_history` | `list[EpochMetrics]` | Per-epoch metrics array (see below) |
| `step_number` | `int` | Current action step iteration |
| `max_steps` | `int` | Limit until terminal state (always 15) |

### EpochMetrics Schema

Each entry in `metrics_history` contains enriched telemetry:

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `int` | Epoch index (0-based) |
| `train_loss` | `float` | Training loss for the epoch |
| `val_loss` | `float` | Validation loss for the epoch |
| `accuracy` | `float \| null` | Classification accuracy (Easy/Medium tasks) |
| `iou` | `float \| null` | Intersection-over-Union score (Hard task) |
| `gpu_memory_allocated_mb` | `int` | Simulated GPU memory allocated in MB |
| `step_time_ms` | `float` | Average simulated time per training step (ms) |
| `gradient_norm` | `float` | Simulated L2 norm of gradients |

### Computed Properties

The observation also exposes convenience properties for agent decision-making:

| Property | Type | Description |
|----------|------|-------------|
| `last_metrics` | `EpochMetrics \| None` | Most recent epoch metrics entry |
| `steps_remaining` | `int` | `max_steps - step_number` |
| `val_train_gap` | `float \| None` | `val_loss - train_loss` from last epoch (useful for medium task) |

---

## 🏆 Reward Function

The reward function provides **dense signal** throughout the workflow — not just sparse end-of-episode scores. Reward is clamped to `[-1.0, 2.0]` per step.

| Event | Reward | Rationale |
|-------|--------|-----------|
| Task SOLVED | **+2.0** | Large bonus for task completion |
| Validation loss drops | **+1.0** per 0.05 drop | High weighting for steady metric improvements |
| Epoch simulated properly | **+0.1** per epoch | Incremental reinforcement for valid continuous execution |
| Valid action but no progress | **−0.01** | Minor detraction to disincentivize idling operations |
| Infinite loop (3× identical action) | **−0.3** | Penalizes agents stuck repeating identical action patterns |
| Crash (OOM / NaN loss / Shape error) | **−0.5** | Strict penalization for chaotic, unchecked changes |

---

## 🚀 Setup & Running Locally

### 1. Clone & Install

```bash
git clone https://github.com/swaraj123822/ML-pipleline-debugger.git
cd ml-pipeline-debugger

# Create a virtual environment
python -m venv venv
source venv/bin/activate         # Linux/Mac
# .\\venv\\Scripts\\activate      # Windows

# Install server + client dependencies
pip install -r server/requirements.txt
pip install -r baseline/requirements.txt
```

### 2. Set Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b:groq"
export HF_TOKEN="your-hf-token"
export MLDBG_BASE_URL="http://localhost:7860"
```

### 3. Start the Environment Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Run the Baseline Agent

```bash
python inference.py --task all
```

### 5. Run Tests

```bash
pytest
```

### 6. Validate Submission

```bash
bash validation-scripts/validate-submission.sh "http://localhost:7860" .
```

---

## 🐳 Docker

```bash
# Build
docker build -t ml-pipeline-debugger .

# Run
docker run -rm -p 7860:7860 ml-pipeline-debugger
```

The environment server initializes instantly with `uvicorn` and binds on port 7860 — no GPU, no model weights, deterministic from first request.

---

## 📡 API Reference

### HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Welcome message with endpoint directory |
| `GET`  | `/health` | Health check (`{"status": "ok"}`) |
| `GET`  | `/info` | Environment metadata (tasks, action space, reward schedule) |
| `POST` | `/reset` | Start a new episode — `?task_id=easy\|medium\|hard` |
| `POST` | `/step` | Submit an action — body: `{"task_id": "...", "action": {...}}` |
| `GET`  | `/state` | Retrieve current episode state |

### WebSocket Protocol (`/ws`)

Persistent session with isolated environment per connection:

```jsonc
// Client → Server
{"method": "reset", "task_id": "easy"}
{"method": "step",  "action": {"action_type": "fix_reshape", "layer": "flatten", "new_shape": [2304]}}
{"method": "state"}

// Server → Client
{"result": <observation | step_result | state>}
{"error":  "<message>"}
```

---

## 🌍 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint for `inference.py` (e.g.`https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Yes | Agent model identifier (e.g. `openai/gpt-4o`, `openai/gpt-oss-120b:groq`) |
| `HF_TOKEN` | Yes | API authentication token |
| `MLDBG_BASE_URL` | No | Overrides localhost targeting server (default: `http://localhost:7860`) |

---


### 🤖 Baseline Agent Results (`openai/gpt-oss-120b:groq`)

| Task | Difficulty | Steps | Score | Passed | Cumulative Reward |
|------|-----------|:-----:|:-----:|:------:|:-----------------:|
| Task 1 — Tensor Shape Error | Easy | 1 | **0.999** | ✅ | 2.0 |
| Task 2 — Overfitting Problems | Medium | 15 | **0.578** | ❌ | 5.956 |
| Task 3 — Segmentation IoU | Hard | 15 | **0.899** | ✅ | 12.0 |
| **Overall Average** | | | **0.825** | | |

---

## 🧪 Test Coverage

The test suite covers **3 modules** with comprehensive edge cases:

| Test Module | Tests | Coverage |
|-------------|:-----:|----------|
| `test_env_loop.py` | 17 | End-to-end episode lifecycle: reset, step, state, termination, reward accumulation |
| `test_graders.py` | 16 | Deterministic grader scoring: partial credits, efficiency penalties, NaN penalties, IoU tiers |
| `test_simulator.py` | 16 | Crash detection, loss curve generation, determinism verification, solve conditions |

```bash
# Run with verbose output
pytest
```

## 📝 OpenEnv Spec Compliance

- ✅ `openenv.yaml` comprehensively declares all 3 tasks, thresholds, and deployment metadata
- ✅ Pydantic v2 strict contracts linking Client ↔ Server (`models.py`) with field-level validation
- ✅ Continuous WebSocket session loops natively supported under `/ws`
- ✅ Dual transport: HTTP `POST /reset`, `POST /step`, `GET /state` alongside WebSocket
- ✅ `step()` → evaluates action against deterministic state machines, outputs observation + reward
- ✅ `reset()` → resets all parameters for precise episode-level tracking
- ✅ `state()` → reflects internal episode checkpoint (action history, cumulative reward, terminal status)
- ✅ **3 progressive tasks** evaluating variable ML failures (CV Dimension Error → Overfitting → IoU Segmentation with hidden trap)
- ✅ **6 typed actions** — 4 functional + 1 distractor + 1 context-dependent (essential for hard, distractor for easy/medium)
- ✅ Enriched `EpochMetrics` with simulated telemetry (`gpu_memory_allocated_mb`, `step_time_ms`, `gradient_norm`)
- ✅ Auto-graded system emitting scores strictly in open interval `(0.001, 0.999)` with efficiency penalties
- ✅ Custom `Dockerfile` ready for Hugging Face Spaces deployment (Python 3.12-slim, port 7860)
- ✅ Dense dynamic reward distribution `[-1.0, 2.0]` with 6 distinct signal components
- ✅ Infinite loop detection (3× repeated identical action → `−0.3` penalty)
- ✅ Baseline `inference.py` supporting `--task` selection and JSON trajectory output
- ✅ Comprehensive test suite: 49 tests across graders, simulator, and environment loop
