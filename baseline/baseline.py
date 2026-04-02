"""
baseline/baseline.py — Baseline Inference Script

Uses the OpenAI API + client.py (WebSocket) to run an LLM agent
against all 3 tasks. Reads OPENAI_API_KEY from environment variables.
Produces a reproducible baseline score written to baseline_results.json.

Usage:
    # Server must be running first:
    #   uvicorn server.app:app --host 0.0.0.0 --port 7860

    OPENAI_API_KEY=sk-... python baseline/baseline.py
    OPENAI_API_KEY=sk-... python baseline/baseline.py --task easy
    OPENAI_API_KEY=sk-... python baseline/baseline.py --model gpt-4o
    OPENAI_API_KEY=sk-... python baseline/baseline.py --url http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from client import MLDebuggerEnv
from models import (
    AddAugmentation,
    AdjustLossWeights,
    FixReshape,
    TuneHyperparameters,
)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_URL = os.environ.get("MLDBG_BASE_URL", "http://localhost:7860")

SYSTEM_PROMPT = """You are an expert ML engineer debugging broken training pipelines.

You receive a JSON observation and must output EXACTLY ONE action as valid JSON.

=== AVAILABLE ACTIONS ===

1. {"action_type": "fix_reshape", "layer": "<name>", "new_shape": [<ints, max 4 dims>]}
2. {"action_type": "tune_hyperparameters", "lr": <float (0,1)>, "batch_size": <power-of-2>, "epochs": <int [1,50]>}
3. {"action_type": "add_augmentation", "strategy": "<dropout|weight_decay|truncate_sequence|horizontal_flip|mixup>"}
4. {"action_type": "adjust_loss_weights", "dice_weight": <float [0,1]>, "ce_weight": <float [0,1]>}
   NOTE: dice_weight + ce_weight MUST equal exactly 1.0

=== TASK HINTS ===
easy:   CNN shape error. Find the correct flatten size from tensor_shapes (conv_out channels x H x W).
medium: Transformer overfitting. Apply 'dropout' then 'truncate_sequence' to close val/train gap < 0.15.
hard:   Lung segmentation. Tune lr to (0.0001–0.003) AND dice_weight to (0.55–0.75) within 5 steps.

Respond with ONLY the raw JSON object. No explanation. No markdown fences."""


def parse_action(text: str) -> Any | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error: {e}")
        return None
    t = raw.get("action_type", "")
    try:
        if t == "fix_reshape":
            return FixReshape(**raw)
        if t == "tune_hyperparameters":
            return TuneHyperparameters(**raw)
        if t == "add_augmentation":
            return AddAugmentation(**raw)
        if t == "adjust_loss_weights":
            return AdjustLossWeights(**raw)
        print(f"  [WARN] Unknown action_type: '{t}'")
        return None
    except Exception as e:
        print(f"  [WARN] Action validation error: {e}")
        return None


def run_task(
    client: OpenAI,
    task_id: Literal["easy", "medium", "hard"],
    model: str,
    base_url: str,
    verbose: bool = True,
) -> dict[str, Any]:
    trajectory: list[dict] = []
    conversation: list[dict] = []

    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'='*60}")

    with MLDebuggerEnv(base_url=base_url).sync() as env:
        obs = env.reset(task_id=task_id)

        for step in range(15):
            user_content = json.dumps(obs.model_dump(mode="json"), indent=2)
            conversation.append({"role": "user", "content": user_content})

            if verbose:
                print(f"\n--- Step {step + 1}/15 ---")
                if obs.error_trace:
                    print(f"  Error: {obs.error_trace[:100]}...")
                if obs.last_metrics:
                    m = obs.last_metrics
                    iou_str = f" iou={m.iou:.4f}" if m.iou else ""
                    print(f"  Metrics: train={m.train_loss:.4f} val={m.val_loss:.4f}{iou_str}")

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *conversation],
                    temperature=0.0,
                    max_tokens=256,
                )
            except Exception as e:
                print(f"  [ERROR] OpenAI call failed: {e}")
                break

            raw_text = response.choices[0].message.content or ""
            conversation.append({"role": "assistant", "content": raw_text})

            if verbose:
                print(f"  LLM: {raw_text[:150]}")

            action = parse_action(raw_text)
            if action is None:
                conversation.append({
                    "role": "user",
                    "content": "Invalid response. Output ONLY a valid JSON action object.",
                })
                continue

            if verbose:
                print(f"  Action: {action.action_type} → {action.model_dump()}")

            result = env.step(action)

            trajectory.append({
                "step": step + 1,
                "action": action.model_dump(),
                "reward": result.reward,
                "cumulative": result.cumulative_reward,
                "reason": result.reward_reason,
                "done": result.done,
                "crash": result.info.get("crash"),
            })

            if verbose:
                print(f"  Reward: {result.reward:+.2f} ({result.reward_reason})")

            obs = result.observation

            if result.done:
                fg = result.info.get("final_grade", {})
                print(f"\n  Done at step {step + 1}.")
                print(f"  Score:  {fg.get('score', 0):.4f}")
                print(f"  Passed: {fg.get('passed', False)}")
                print(f"  Detail: {fg.get('reason', '')}")
                return {
                    "task_id": task_id, "model": model,
                    "steps_taken": step + 1,
                    "score": fg.get("score", 0.0),
                    "passed": fg.get("passed", False),
                    "cumulative_reward": result.cumulative_reward,
                    "trajectory": trajectory,
                    "grade_detail": fg,
                }

            time.sleep(0.3)

        # Hit step limit
        final_state = env.state()
        print(f"\n  Max steps reached.")
        return {
            "task_id": task_id, "model": model,
            "steps_taken": 15,
            "score": 0.0,
            "passed": False,
            "cumulative_reward": final_state.cumulative_reward,
            "trajectory": trajectory,
            "grade_detail": {},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Pipeline Debugger — Baseline Agent")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    parser.add_argument("--output", default="baseline_results.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    oai = OpenAI(api_key=api_key)
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results: list[dict] = []

    for task_id in tasks:
        r = run_task(oai, task_id, args.model, args.url, verbose=not args.quiet)
        results.append(r)

    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<10} {'Score':>8} {'Passed':>8} {'Steps':>7} {'Cum.Reward':>12}")
    print("-" * 50)

    total = 0.0
    for r in results:
        print(
            f"{r['task_id']:<10} {r['score']:>8.4f} "
            f"{'YES' if r['passed'] else 'NO':>8} "
            f"{r['steps_taken']:>7} {r['cumulative_reward']:>12.4f}"
        )
        total += r["score"]

    avg = total / len(results) if results else 0.0
    print("-" * 50)
    print(f"{'AVERAGE':<10} {avg:>8.4f}")
    print(f"\nModel: {args.model}")

    output = {"model": args.model, "average_score": avg, "tasks": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()