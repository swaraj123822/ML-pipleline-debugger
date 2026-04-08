import argparse
import json
import os
import re
import sys
import time
from typing import Any, Literal

from openai import OpenAI

from client import MLDebuggerEnv
from models import (
    AddAugmentation,
    AdjustLossWeights,
    FixReshape,
    TuneHyperparameters,
    ChangeOptimizer,
    ToggleLayerFreeze,
)

API_BASE_URL = os.environ.get("API_BASE_URL","https://openrouter.ai/api/v1")
MODEL_NAME = os.environ.get("MODEL_NAME","openai/gpt-oss-120b:free")
ENV_URL = os.environ.get("MLDBG_BASE_URL", "http://localhost:7860")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

SYSTEM_PROMPT = """You are an expert ML engineer debugging broken training pipelines.

You receive a JSON observation and must output EXACTLY ONE action as valid JSON.

=== AVAILABLE ACTIONS ===

1. {"action_type": "fix_reshape", "layer": "<name>", "new_shape": [<ints, max 4 dims>]}
2. {"action_type": "tune_hyperparameters", "lr": <float (0,1)>, "batch_size": <power-of-2>, "epochs": <int [1,50]>}
3. {"action_type": "add_augmentation", "strategy": "<dropout|weight_decay|truncate_sequence|horizontal_flip|mixup>"}
4. {"action_type": "adjust_loss_weights", "dice_weight": <float [0,1]>, "ce_weight": <float [0,1]>}
   NOTE: dice_weight + ce_weight MUST equal exactly 1.0
5. {"action_type": "change_optimizer", "optimizer": "<Adam|SGD|RMSprop>", "weight_decay": <float>}
6. {"action_type": "toggle_layer_freeze", "layer_name": "<name>", "freeze": <true|false>}

=== TASK HINTS ===
easy:   CNN shape error. Find the correct flatten size from tensor_shapes (conv_out channels x H x W).
medium: Transformer overfitting. Apply 'dropout' then 'truncate_sequence' to close val/train gap < 0.15.
hard:   Lung segmentation. Tune lr to (0.0001–0.003) AND dice_weight to (0.55–0.75) within 5 steps.

Respond with ONLY the raw JSON object. No explanation. No markdown fences."""


def parse_action(text: str) -> Any | None:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
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
        if t == "change_optimizer":
            return ChangeOptimizer(**raw)
        if t == "toggle_layer_freeze":
            return ToggleLayerFreeze(**raw)
        return None
    except Exception:
        return None


def run_task(
    client: OpenAI,
    task_id: Literal["easy", "medium", "hard"],
    model: str,
    base_url: str,
) -> dict[str, Any]:
    trajectory: list[dict] = []
    conversation: list[dict] = []

    print(f"[START] task={task_id} env=ml-pipeline-debugger model={model}", flush=True)

    with MLDebuggerEnv(base_url=base_url).sync() as env:
        obs = env.reset(task_id=task_id)

        for step in range(15):
            user_content = json.dumps(obs.model_dump(mode="json"), indent=2)
            conversation.append({"role": "user", "content": user_content})

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *conversation],
                    temperature=0.0,
                    max_tokens=256,
                )
            except Exception as e:
                print(f"[STEP] step={step + 1} action=null reward=0.00 done=false error=api_error", flush=True)
                break

            raw_text = response.choices[0].message.content or ""
            conversation.append({"role": "assistant", "content": raw_text})

            action = parse_action(raw_text)
            if action is None:
                print(f"[STEP] step={step + 1} action=null reward=0.00 done=false error=invalid_json", flush=True)
                conversation.append({
                    "role": "user",
                    "content": "Invalid response. Output ONLY a valid JSON action object. No text before or after.",
                })
                continue

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

            error_val = result.info.get("crash", {}).get("type") if result.info.get("crash") else "null"
            done_val = "true" if result.done else "false"
            action_str = json.dumps(action.model_dump())
            print(f"[STEP] step={step + 1} action={action_str} reward={result.reward:.3f} done={done_val} error={error_val}", flush=True)

            obs = result.observation

            if result.done:
                fg = result.info.get("final_grade", {})
                score = fg.get("score", 0.0)
                score = max(0.001, min(0.999, score))
                success_val = "true" if fg.get("passed", False) else "false"
                rewards_str = ",".join(f"{r['reward']:.2f}" for r in trajectory)
                print(f"[END] success={success_val} steps={step + 1} score={score:.3f} rewards={rewards_str}", flush=True)
                return {
                    "task_id": task_id,
                    "model": model,
                    "steps_taken": step + 1,
                    "score": score,
                    "passed": fg.get("passed", False),
                    "cumulative_reward": result.cumulative_reward,
                    "trajectory": trajectory,
                    "grade_detail": fg,
                }

            time.sleep(0.3)

        final_state = env.state()
        rewards_str = ",".join(f"{r['reward']:.2f}" for r in trajectory)
        print(f"[END] success=false steps=15 score=0.001 rewards={rewards_str}", flush=True)
        return {
            "task_id": task_id,
            "model": model,
            "steps_taken": 15,
            "score": 0.001,
            "passed": False,
            "cumulative_reward": final_state.cumulative_reward,
            "trajectory": trajectory,
            "grade_detail": {},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Pipeline Debugger — Baseline Agent")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--output", default="baseline_results.json")
    args = parser.parse_args()

    if not HF_TOKEN or not API_BASE_URL or not MODEL_NAME:
        sys.exit(1)

    oai = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results: list[dict] = []

    for task_id in tasks:
        r = run_task(oai, task_id, MODEL_NAME, ENV_URL)
        results.append(r)

    avg = sum(r["score"] for r in results) / len(results) if results else 0.001

    output = {"model": MODEL_NAME, "average_score": avg, "tasks": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()