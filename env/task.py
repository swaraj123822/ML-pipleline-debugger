"""
task module defines the core logic for ML debugging tasks in the environment.

Each task defines:
  - An initial Observation the agent receives on reset()
  - A deterministic grader that scores agent performance 0.0 -> 1.0
  - Clear success/failure criteria

Tasks:
  easy   — CV Shape Error         (fix tensor reshape mismatch)
  medium — NLP Overfitting        (apply augmentation to close val gap)
  hard   — Lung Segmentation IoU  (tune lr + loss weights to push IoU > 0.85)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from env.models import EpochMetrics, Observation, TaskState


# ---------------------------------------------------------------------------
# Grader result
# ---------------------------------------------------------------------------


@dataclass
class GraderResult:
    """Output of a task grader. Score is always in [0.0, 1.0]."""
    score: float          # 0.0 = failed, 1.0 = perfect
    passed: bool          # True if score >= success_threshold
    reason: str           # Human-readable explanation
    partial_credit: dict[str, float]  # Breakdown of score components


# ---------------------------------------------------------------------------
# Task definitions — initial observations
# ---------------------------------------------------------------------------


def get_initial_observation(task_id: Literal["easy", "medium", "hard"]) -> Observation:
    """Return the starting Observation for a given task."""
    if task_id == "easy":
        return _initial_obs_easy()
    if task_id == "medium":
        return _initial_obs_medium()
    if task_id == "hard":
        return _initial_obs_hard()
    raise ValueError(f"Unknown task_id: {task_id}")


def _initial_obs_easy() -> Observation:
    return Observation(
        task_id="easy",
        architecture_summary=(
            "CNN (BROKEN): Conv2d(3,64,3,pad=0) -> ReLU -> MaxPool2d(5) -> "
            "Flatten([512]) <- WRONG -> Linear(512,10)\n"
            "The Flatten layer is configured with the wrong output size. "
            "The Conv2d output after MaxPool2d is [batch, 64, 6, 6] = 2304 elements, "
            "but Flatten is set to produce only 512 elements."
        ),
        tensor_shapes={
            "input": [1, 3, 32, 32],
            "conv1_out": [1, 64, 30, 30],
            "pool_out": [1, 64, 6, 6],
            "flatten_out": [1, 512],        # BUG: should be [1, 2304]
            "fc1_in_expected": [1, 2304],   # What Linear(2304,512) expects
        },
        error_trace=(
            "RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x512 and 2304x512).\n"
            "  File 'model.py', line 23, in forward\n"
            "    x = self.fc1(x.view(x.size(0), -1))\n"
            "torch.nn.Linear weight shape: [512, 2304]\n"
            "Received input shape: [1, 512]\n"
            "Hint: Check the output dimensions of your Conv2d and MaxPool2d layers."
        ),
        metrics_history=[],
        step_number=0,
        max_steps=15,
    )


def _initial_obs_medium() -> Observation:
    return Observation(
        task_id="medium",
        architecture_summary=(
            "Transformer: Embedding(30522,256) -> 4xTransformerEncoderLayer(d=256, heads=8) "
            "-> Linear(256,2) [binary sentiment classifier]\n"
            "Dataset: 10,000 samples with severe class imbalance (95% negative, 5% positive).\n"
            "Problem: Model memorizes training data by epoch 2. Validation loss diverges "
            "immediately while training loss keeps dropping."
        ),
        tensor_shapes={
            "input_ids": [16, 512],
            "embedding_out": [16, 512, 256],
            "encoder_out": [16, 512, 256],
            "cls_token": [16, 256],
            "logits": [16, 2],
        },
        error_trace=None,
        metrics_history=[
            EpochMetrics(epoch=0, train_loss=1.80, val_loss=2.40, accuracy=0.47),
            EpochMetrics(epoch=1, train_loss=0.95, val_loss=2.85, accuracy=0.51),
        ],
        step_number=0,
        max_steps=15,
    )


def _initial_obs_hard() -> Observation:
    return Observation(
        task_id="hard",
        architecture_summary=(
            "SegNet: ResNet50-Encoder -> AdaptiveAttentionGate(512) -> "
            "LightweightDecoder(256,128,64) -> Sigmoid [5-class lung segmentation]\n"
            "Cross-dataset setup: trained on ChestX-ray14, validated on RSNA Pneumonia.\n"
            "Current loss: 0.6 * CrossEntropy + 0.4 * Dice  (suboptimal balance).\n"
            "Problem: IoU plateaus at 0.52 — need to exceed 0.85 within 5 steps."
        ),
        tensor_shapes={
            "input": [2, 3, 512, 512],
            "encoder_out": [2, 512, 16, 16],
            "attention_out": [2, 512, 16, 16],
            "decoder_out": [2, 5, 512, 512],
            "prediction": [2, 5, 512, 512],
        },
        error_trace=None,
        metrics_history=[
            EpochMetrics(epoch=0, train_loss=0.85, val_loss=0.91, iou=0.52),
        ],
        step_number=0,
        max_steps=15,
    )


# ---------------------------------------------------------------------------
# Graders — deterministic scoring 0.0 -> 1.0
# ---------------------------------------------------------------------------


def grade(task_id: Literal["easy", "medium", "hard"], state: TaskState) -> GraderResult:
    """
    Score the agent's final performance on a task.
    Called at episode end (done=True) by the environment.

    All graders are deterministic — same state always produces same score.
    """
    if task_id == "easy":
        return _grade_easy(state)
    if task_id == "medium":
        return _grade_medium(state)
    if task_id == "hard":
        return _grade_hard(state)
    raise ValueError(f"Unknown task_id: {task_id}")


def _grade_easy(state: TaskState) -> GraderResult:
    """
    Easy grader — CV Shape Error.

    Scoring:
      1.0  — Correct FixReshape submitted (layer=flatten, new_shape=[2304])
      0.4  — Correct layer identified but wrong shape
      0.2  — FixReshape attempted on wrong layer
      0.0  — No FixReshape attempted at all
    Efficiency bonus: full marks in ≤ 2 steps → score stays 1.0.
    Penalty: each wasted step past 2 → -0.05 (floor 0.7 if eventually correct).
    """
    from env.models import FixReshape  # local import to avoid circular

    fix_attempts = [a for a in state.action_history if isinstance(a, FixReshape)]

    partial: dict[str, float] = {
        "fix_attempted": 0.0,
        "correct_layer": 0.0,
        "correct_shape": 0.0,
        "efficiency": 0.0,
    }

    if not fix_attempts:
        return GraderResult(
            score=0.0,
            passed=False,
            reason="Agent never submitted a FixReshape action.",
            partial_credit=partial,
        )

    partial["fix_attempted"] = 0.2

    # Check best attempt
    correct_layer_seen = any(a.layer == "flatten" for a in fix_attempts)
    correct_fix_seen = any(
        a.layer == "flatten" and a.new_shape == [2304] for a in fix_attempts
    )

    if correct_layer_seen:
        partial["correct_layer"] = 0.2

    if correct_fix_seen:
        partial["correct_shape"] = 0.6
        # Efficiency: find step index of correct fix
        correct_step = next(
            i for i, a in enumerate(state.action_history)
            if isinstance(a, FixReshape) and a.layer == "flatten" and a.new_shape == [2304]
        )
        wasted = max(0, correct_step - 1)
        partial["efficiency"] = max(0.0, 0.2 - wasted * 0.05)

        score = sum(partial.values())
        return GraderResult(
            score=min(1.0, round(score, 4)),
            passed=True,
            reason=f"Correct fix applied at step {correct_step}. "
                   f"Efficiency penalty: {wasted} wasted steps.",
            partial_credit=partial,
        )

    score = sum(partial.values())
    return GraderResult(
        score=round(score, 4),
        passed=False,
        reason=f"FixReshape attempted but shape never correct. "
               f"Correct layer identified: {correct_layer_seen}.",
        partial_credit=partial,
    )


def _grade_medium(state: TaskState) -> GraderResult:
    """
    Medium grader — NLP Overfitting.

    Scoring:
      0.3  — At least one relevant strategy applied (dropout or truncate_sequence)
      0.3  — Both required strategies applied
      0.4  — Final val_loss - train_loss < 0.15
    Penalty: -0.1 if agent caused a NaN crash at any point.
    """
    from env.models import AddAugmentation, TuneHyperparameters  # local import

    partial: dict[str, float] = {
        "strategy_started": 0.0,
        "both_strategies": 0.0,
        "gap_closed": 0.0,
    }

    applied: set[str] = set()
    caused_nan = False

    for a in state.action_history:
        if isinstance(a, AddAugmentation):
            applied.add(a.strategy)
        if isinstance(a, TuneHyperparameters) and a.lr > 0.5:
            caused_nan = True

    required = {"dropout", "truncate_sequence"}
    relevant_applied = applied & required

    if relevant_applied:
        partial["strategy_started"] = 0.3
    if required.issubset(applied):
        partial["both_strategies"] = 0.3

    # Check final metrics gap
    gap_closed = False
    if state.last_observation and state.last_observation.last_metrics:
        m = state.last_observation.last_metrics
        gap = m.val_loss - m.train_loss
        if gap < 0.15 and relevant_applied:
            partial["gap_closed"] = 0.4
            gap_closed = True

    score = sum(partial.values())
    if caused_nan:
        score = max(0.0, score - 0.1)

    passed = score >= 0.8
    reason_parts = []
    if relevant_applied:
        reason_parts.append(f"Applied strategies: {sorted(applied)}")
    else:
        reason_parts.append("No relevant strategies applied")
    if gap_closed:
        reason_parts.append("Validation gap closed successfully")
    if caused_nan:
        reason_parts.append("NaN crash penalty applied")

    return GraderResult(
        score=round(min(1.0, score), 4),
        passed=passed,
        reason=". ".join(reason_parts) + ".",
        partial_credit=partial,
    )


def _grade_hard(state: TaskState) -> GraderResult:
    """
    Hard grader — Lung Segmentation IoU.

    Scoring is based on best IoU achieved within the step limit:
      IoU >= 0.85 within 5 steps  → 1.0
      IoU >= 0.85 but > 5 steps   → 0.7
      IoU >= 0.75                  → 0.5
      IoU >= 0.65                  → 0.3
      IoU >= 0.55                  → 0.15
      IoU < 0.55                   → 0.0

    Partial credit for config quality (lr and loss weights in optimal range).
    """
    from env.models import AdjustLossWeights, TuneHyperparameters  # local import

    partial: dict[str, float] = {
        "iou_achievement": 0.0,
        "config_quality": 0.0,
    }

    # Gather best IoU from history
    best_iou = 0.52  # starting value
    if state.last_observation:
        for m in state.last_observation.metrics_history:
            if m.iou is not None:
                best_iou = max(best_iou, m.iou)

    # IoU achievement score
    achieved_within_limit = state.current_step <= 5 and best_iou >= 0.85

    if best_iou >= 0.85:
        partial["iou_achievement"] = 1.0 if achieved_within_limit else 0.7
    elif best_iou >= 0.75:
        partial["iou_achievement"] = 0.5
    elif best_iou >= 0.65:
        partial["iou_achievement"] = 0.3
    elif best_iou >= 0.55:
        partial["iou_achievement"] = 0.15

    # Config quality partial credit
    best_lr = None
    best_dice = None
    for a in state.action_history:
        if isinstance(a, TuneHyperparameters):
            best_lr = a.lr
        if isinstance(a, AdjustLossWeights):
            best_dice = a.dice_weight

    config_score = 0.0
    if best_lr is not None and 1e-4 <= best_lr <= 3e-3:
        config_score += 0.1
    if best_dice is not None and 0.55 <= best_dice <= 0.75:
        config_score += 0.1
    partial["config_quality"] = min(0.2, config_score)

    # Total — cap at 1.0, but iou_achievement is the primary gate
    raw_score = partial["iou_achievement"] + (
        partial["config_quality"] if partial["iou_achievement"] < 1.0 else 0.0
    )
    score = min(1.0, round(raw_score, 4))
    passed = score >= 0.85

    return GraderResult(
        score=score,
        passed=passed,
        reason=(
            f"Best IoU: {best_iou:.4f} "
            f"({'within' if achieved_within_limit else 'exceeded'} 5-step limit). "
            f"Config quality: lr={best_lr}, dice_weight={best_dice}."
        ),
        partial_credit=partial,
    )


# ---------------------------------------------------------------------------
# Task metadata registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, dict] = {
    "easy": {
        "id": "easy",
        "name": "CV Shape Error",
        "difficulty": "easy",
        "success_threshold": 1.0,
        "max_steps": 15,
        "description": (
            "Fix a tensor reshape mismatch in a CNN pipeline. "
            "The agent must identify that Flatten produces 512 features "
            "when 2304 are required, and submit the correct FixReshape action."
        ),
    },
    "medium": {
        "id": "medium",
        "name": "NLP Overfitting",
        "difficulty": "medium",
        "success_threshold": 0.8,
        "max_steps": 15,
        "description": (
            "Resolve catastrophic overfitting in a Transformer classifier. "
            "The agent must apply dropout and sequence truncation to close "
            "the train/val loss gap below 0.15."
        ),
    },
    "hard": {
        "id": "hard",
        "name": "Lung Segmentation IoU",
        "difficulty": "hard",
        "success_threshold": 0.85,
        "max_steps": 15,
        "description": (
            "Optimize a cross-dataset lung segmentation model to exceed IoU 0.85 "
            "within 5 steps by tuning the learning rate scheduler and "
            "rebalancing Dice vs Cross-Entropy loss weights."
        ),
    },
}



print(" task.py  loaded successfully")