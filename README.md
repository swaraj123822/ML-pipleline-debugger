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

[🚀 Live Demo](https://huggingface.co/spaces/subhrajit36/my-env) · [📄 Docs](#-api-reference) · [🐛 Issues](https://github.com/swaraj123822/ML-pipleline-debugger/issues)

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
│              FastAPI WebSocket Server  (port 7860)          │
│                                                             │
│   ┌─────────────────┐        ┌──────────────────────────┐  │
│   │  Action Router  │──────▶ │   Simulation Engine      │  │
│   │  (Pydantic v2)  │        │   (<50ms per step)       │  │
│   └─────────────────┘        └──────────┬───────────────┘  │
│                                         │                   │
│                              ┌──────────▼───────────────┐  │
│                              │   JSON Scenario Files     │  │
│                              │  ┌─────────────────────┐ │  │
│                              │  │ easy_cv_shapes.json  │ │  │
│                              │  │ medium_nlp_overfit.. │ │  │
│                              │  │ hard_lung_seg.json   │ │  │
│                              │  └─────────────────────┘ │  │
│                              └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │  Observation (error_msg, metrics,
                            │  tensor_shapes, reward, done)
                            ▼
                       AI AGENT
```

No real model training occurs at any point. Every response — error messages, training logs, metrics — is authored in advance and stored in JSON files. This design allows the environment to be fully deterministic and GPU-free.

---

## 🗂️ Project Structure

```
ml-pipeline-debugger/
├── server/
│   ├── app.py            # FastAPI server (Endpoints + WebSocket /ws)
│   ├── environment.py    # Core OpenEnv MLDebuggerEnvironment logic
│   ├── simulator.py      # CPU-only simulation engine assessing crashes & metrics
│   ├── tasks.py          # 3 difficulty tasks and determinstic auto-graders
│   └── reward.py         # Dense step-by-step reward configuration
├── validation-scripts/
│   └── validate-submission.sh # Utility to validate OpenEnv submission criteria
├── models.py             # Pydantic Action/Observation/State structures
├── inference.py          # Baseline LLM agent using OpenAI compatibility
├── client.py             # Reusable Python Websocket Client (MLDebuggerEnv)
├── openenv.yaml          # OpenEnv metadata & spec
├── Dockerfile            # Docker container config
└── README.md             # This file
```

---

## 🎯 Tasks

### Task 1 — Easy: CV Shape Error
- **Problem:** A CNN's Conv layers output the wrong shape (512 vs 2304) for the subsequent Linear layer, crashing at forward pass with a dimension mismatch.
- **Correct action:** `fix_reshape` with the right flatten dimension (`layer="flatten"`, `new_shape=[2304]`).
- **Grader:** Evaluates both correct diagnosis of the layer and supplying the precise shape dimensions.
- **Max steps:** 15

### Task 2 — Medium: NLP Overfitting
- **Problem:** A Transformer text classifier is memorizing training data. Train loss plummets drastically, but validation loss rapidly diverges and spikes.
- **Correct actions:** `add_augmentation` applying both `dropout` and `truncate_sequence` sequentially to stabilize the validation gap below 0.15.
- **Grader:** Credits correctly started regularization strategies and heavily rewards closing the val/train loss gap metric safely.
- **Max steps:** 15

### Task 3 — Hard: Lung Segmentation IoU
- **Problem:** A U-Net segmentation architecture is plateaued at IoU=0.52 due to unbalanced Cross-Entropy/Dice losses and suboptimal learning rates. Target is `>0.85`.
- **Correct actions:** `adjust_loss_weights` (balance Dice weight between 0.55-0.75) + `tune_hyperparameters` (locate optimal learning rate between 0.0001–0.003).
- **Grader:** Measures if IoU target is reached inside the strict 5-step limit, scoring learning rate and dice optimality.
- **Max steps:** 15

---

## 🔌 Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `fix_reshape` | `layer: str, new_shape: list[int]` | Fix dimension mismatch in Linear/Flatten layers |
| `tune_hyperparameters` | `lr: float, batch_size: int, epochs: int` | Adjust training hyperparameters seamlessly |
| `add_augmentation` | `strategy: str` (`dropout`, `weight_decay`, `truncate_sequence`, `horizontal_flip`, `mixup`) | Apply regularization / data augmentation |
| `adjust_loss_weights` | `dice_weight: float, ce_weight: float` | Balance target losses (Must equal 1.0) |
| `change_optimizer` | `optimizer: str` (`Adam`, `SGD`, `RMSprop`), `weight_decay: float` | Distractor action: Switch optimizer configurations |
| `toggle_layer_freeze` | `layer_name: str, freeze: bool` | Distractor action: Freeze gradients in specific model layers |

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task marker (`easy`, `medium`, `hard`) |
| `architecture_summary` | `str` | Text description describing module composition & layers |
| `tensor_shapes` | `dict[str, list[int]]` | Expected vs actual input tensor dimension states |
| `error_trace` | `str` (or `null`) | Python Stacktrace simulation indicating Crash events |
| `metrics_history` | `list[EpochMetrics]` | Array containing `train_loss`, `val_loss`, `accuracy`, and `iou` |
| `step_number` | `int` | Current action step iteration |
| `max_steps` | `int` | Limits until terminal configuration (Always 15) |

---

## 🏆 Reward Function

The reward function provides **dense signal** throughout the model workflow — not just sparse end-of-episode scores.

| Event | Reward | Rationale |
|-------|--------|-----------|
| Task SOLVED | +2.0 | Standard big bonus for task completion |
| Validation loss drops | +1.0 per 0.05 drop | High weighting for steady metric improvements over steps |
| Epoch simulated properly | +0.1 per epoch | Incremental reinforcement emphasizing valid, continuous execution |
| Valid action but no progress | -0.01 | Minor detraction to disincentivize idling operations |
| Infinite Loop (Repeated failure) | -0.3 | Halts models stuck repeating identical syntax patterns |
| Crash (OOM / NaN loss / Error) | -0.5 | Strict penalization minimizing chaotic, unchecked changes |

---

## 🚀 Setup & Running Locally

### 1. Clone & Install Server/Client Requirements

```bash
git clone <your_repo_url>
cd ml-pipeline-debugger

# Create a virtual environment
python -m venv venv
source venv/Scripts/activate # On Windows: .\venv\Scripts\activate

# Install core environment + client endpoints
pip install -r server/requirements.txt
pip install -r baseline/requirements.txt
```

### 2. Set Environment Variables

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="openai/gpt-4o"
export HF_TOKEN="your-hf-token"
export MLDBG_BASE_URL="http://localhost:7860"
```

### 3. Start the Environment Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Run the Baseline Agent

```bash
# Verify inference across all 3 difficulties directly via WebSocket
python inference.py --task all
```

### 5. Validate Submission Standard

```bash
bash validation-scripts/validate-submission.sh "http://localhost:7860" .
```

---

## 🐳 Docker

```bash
# Build
docker build -t ml-pipeline-debugger .

# Run
docker run -p 7860:7860 ml-pipeline-debugger
```

The environment server initializes instantly with `uvicorn` and will bind on port 7860 natively.

---

## 🌍 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint for `inference.py` (e.g. `https://openrouter.ai/api/v1`) |
| `MODEL_NAME` | Yes | Agent model identifier (e.g. `openai/gpt-oss-120b:free`) |
| `HF_TOKEN` | Yes | API authentication token. |
| `MLDBG_BASE_URL` | No | Overrides localhost bindings targeting server (default: `http://localhost:7860`) |

---

## 📊 Baseline Scores

Evaluated reliably interacting with `openai/gpt-oss-120b:free`.

| Task | Difficulty | Score | Passed | Cumulative Reward |
|------|-----------|-------|--------|-------------------|
| Task 1 — CV Shape Error | Easy | **0.999** | ✅ True | 2.0 |
| Task 2 — NLP Overfitting | Medium| **0.600** | ❌ False| 12.922 |
| Task 3 — Lung Segmentation IoU| Hard | **0.999** | ✅ True | 4.0 |
| **Overall Average Score** | | **0.866** | | |

*Note: Models can dynamically struggle with Task 2's specific multi-step requirement (applying Dropout AND Sequence Truncation consecutively to clear the constraint gap thresholds).*

---

## 📝 OpenEnv Spec Compliance

- ✅ `openenv.yaml` comprehensively integrated linking the deployment.
- ✅ Pydantic-based deterministic inputs linking Client `<->` Server. (`models.py`)
- ✅ Continuous WebSocket session loops natively supported under `/ws`.
- ✅ `step()` -> evaluates action correctly against deterministic state machines, outputting info & reward structures.
- ✅ `reset()` -> resets local parameters establishing precise tracking constraints.
- ✅ `state()` -> reflects overarching debug pipeline checkpoints.
- ✅ **3 robust tasks** evaluating variable ML failures (CV Dimension Error, Overfitting Text Classification, IoU Segmentation).
- ✅ Auto-graded system emitting scores scaled cleanly to bounds strictly `[0.0, 1.0]`.
- ✅ Custom `Dockerfile` ready explicitly for Hugging Face Spaces integration.
- ✅ Baseline `inference.py` evaluating `gpt-*` variants effectively utilizing the environment definitions cleanly.
- ✅ Dense dynamic reward distribution allocating partial credits safely preventing sparsity hurdles.
