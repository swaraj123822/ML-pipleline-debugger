---
title: ML Pipeline Debugger
emoji: 🐛
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# ML Pipeline Debugger

**OpenEnv Environment** for the Reinforcement Learning Hackathon 2026

An AI agent acts as an ML Engineer debugging and optimizing broken deep learning pipelines (Computer Vision + NLP).  
The environment uses a **fast simulation engine** so every `step()` runs in <50 ms — no real training required.

## Status
- OpenEnv spec compliant
- 3 tasks (Easy → Medium → Hard) implemented
- Dense reward + programmatic graders


# ML Pipeline Debugger

**OpenEnv Environment** for the Reinforcement Learning Hackathon 2026

An AI agent acts as an ML Engineer debugging and optimizing broken deep learning pipelines (Computer Vision + NLP).  
The environment uses a **fast simulation engine** so every `step()` runs in <50 ms — no real training required.

## Status
- First commit (setup phase)
- Full OpenEnv spec compliance in progress
- 3 tasks (Easy → Medium → Hard) planned
- Dense reward + programmatic graders coming soon

## Quick Start (after full implementation)

```bash
# 1. Clone & setup
git clone https://github.com/YOUR_USERNAME/ml-pipeline-debugger.git
cd ml-pipeline-debugger
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. Install
pip install -e .

# 3. Run baseline (when ready)
OPENAI_API_KEY=sk-... python baseline/baseline.py