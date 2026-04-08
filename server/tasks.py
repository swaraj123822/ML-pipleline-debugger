"""
server/tasks.py — Three Themed Tasks & Deterministic Graders

Each task provides:
  - get_initial_observation()  → starting MLDebuggerObservation
  - grade(state)               → GraderResult with score 0.0–1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from models import (
    AddAugmentation,
    AdjustLossWeights,
    EpochMetrics,
    FixReshape,
    MLDebuggerObservation,
    MLDebuggerState,
    TuneHyperparameters,
)


@dataclass
class GraderResult:
    score: float
    passed: bool
    reason: str
    partial_credit: dict[str, float]
    

def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1) as required by the validator."""
    return round(max(0.001, min(0.999, score)), 3)


# ---------------------------------------------------------------------------
# Initial observations
# ---------------------------------------------------------------------------

def get_initial_observation(task_id: Literal["easy", "medium", "hard"]) -> MLDebuggerObservation:
    if task_id == "easy":
        return MLDebuggerObservation(
            task_id="easy",
            architecture_summary=(
                "CNN (BROKEN): Conv2d(3,64,3,pad=0)->ReLU->MaxPool2d(5)->"
                "Flatten([512])<-WRONG->Linear(512,10)\n"
                "The Flatten layer outputs 512 features but Linear expects 2304 (64x6x6)."
            ),
            tensor_shapes={
                "input": [1, 3, 32, 32],
                "conv1_out": [1, 64, 30, 30],
                "pool_out": [1, 64, 6, 6],
                "flatten_out": [1, 512],
                "fc1_in_expected": [1, 2304],
            },
            error_trace=(
                "RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x512 and 2304x512).\n"
                "  File 'model.py', line 23, in forward\n"
                "    x = self.fc1(x.view(x.size(0), -1))\n"
                "Hint: Check the output dimensions of Conv2d and MaxPool2d."
            ),
            metrics_history=[],
            step_number=0,
            max_steps=15,
        )

    if task_id == "medium":
        return MLDebuggerObservation(
            task_id="medium",
            architecture_summary=(
                "Transformer: Embedding(30522,256)->4xTransformerEncoderLayer(d=256,heads=8)"
                "->Linear(256,2) [binary classifier, 95%/5% class imbalance]\n"
                "Problem: Model memorizes training data by epoch 2. Val loss diverges."
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

    # hard
    return MLDebuggerObservation(
        task_id="hard",
        architecture_summary=(
            "SegNet: ResNet50-Encoder->AdaptiveAttentionGate(512)->"
            "LightweightDecoder(256,128,64)->Sigmoid [5-class lung segmentation]\n"
            "Cross-dataset: trained ChestX-ray14, validated RSNA Pneumonia.\n"
            "Current loss: 0.6*CrossEntropy + 0.4*Dice. IoU plateaus at 0.52. Target: >0.85 in 5 steps."
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
# Graders
# ---------------------------------------------------------------------------

def grade(task_id: Literal["easy", "medium", "hard"], state: MLDebuggerState) -> GraderResult:
    if task_id == "easy":
        return _grade_easy(state)
    if task_id == "medium":
        return _grade_medium(state)
    if task_id == "hard":
        return _grade_hard(state)
    raise ValueError(f"Unknown task_id: {task_id}")


def _grade_easy(state: MLDebuggerState) -> GraderResult:
    fix_attempts = [a for a in state.action_history if isinstance(a, FixReshape)]
    partial = {"fix_attempted": 0.0, "correct_layer": 0.0, "correct_shape": 0.0, "efficiency": 0.0}

    if not fix_attempts:
        return GraderResult(score=0.001, passed=False, reason="No FixReshape action submitted.", partial_credit=partial)

    partial["fix_attempted"] = 0.2
    correct_layer_seen = any(a.layer == "flatten" for a in fix_attempts)
    correct_fix_seen = any(a.layer == "flatten" and a.new_shape == [2304] for a in fix_attempts)

    if correct_layer_seen:
        partial["correct_layer"] = 0.2
    if correct_fix_seen:
        partial["correct_shape"] = 0.6
        correct_step = next(
            i for i, a in enumerate(state.action_history)
            if isinstance(a, FixReshape) and a.layer == "flatten" and a.new_shape == [2304]
        )
        wasted = max(0, correct_step - 1)
        partial["efficiency"] = max(0.0, 0.2 - wasted * 0.05)
        return GraderResult(
            score=_clamp(sum(partial.values())),
            passed=True,
            reason=f"Correct fix at step {correct_step}. Wasted steps: {wasted}.",
            partial_credit=partial,
        )

    return GraderResult(
        score=_clamp(sum(partial.values())),
        passed=False,
        reason=f"FixReshape attempted but shape incorrect. Correct layer identified: {correct_layer_seen}.",
        partial_credit=partial,
    )


def _grade_medium(state: MLDebuggerState) -> GraderResult:
    partial = {"strategy_started": 0.0, "both_strategies": 0.0, "gap_closed": 0.0}
    applied: set[str] = set()
    caused_nan = False

    for a in state.action_history:
        if isinstance(a, AddAugmentation):
            applied.add(a.strategy)
        if isinstance(a, TuneHyperparameters) and a.lr > 0.5:
            caused_nan = True

    required = {"dropout", "truncate_sequence"}
    if applied & required:
        partial["strategy_started"] = 0.3
    if required.issubset(applied):
        partial["both_strategies"] = 0.3

    gap_closed = False
    if state.last_observation and state.last_observation.last_metrics:
        m = state.last_observation.last_metrics
        if m.val_loss - m.train_loss < 0.15 and applied & required:
            partial["gap_closed"] = 0.4
            gap_closed = True

    score = sum(partial.values())
    if caused_nan:
        score = max(0.0, score - 0.1)

    reasons = []
    if applied & required:
        reasons.append(f"Applied: {sorted(applied)}")
    else:
        reasons.append("No relevant strategies applied")
    if gap_closed:
        reasons.append("Val gap closed")
    if caused_nan:
        reasons.append("NaN crash penalty")

    return GraderResult(
        score=_clamp(score),
        passed=score >= 0.8,
        reason=". ".join(reasons) + ".",
        partial_credit=partial,
    )


def _grade_hard(state: MLDebuggerState) -> GraderResult:
    partial = {"iou_achievement": 0.0, "config_quality": 0.0}

    best_iou = 0.52
    if state.last_observation:
        for m in state.last_observation.metrics_history:
            if m.iou is not None:
                best_iou = max(best_iou, m.iou)

    achieved_within_limit = state.current_step <= 5 and best_iou >= 0.85

    if best_iou >= 0.85:
        partial["iou_achievement"] = 0.999 if achieved_within_limit else 0.7
    elif best_iou >= 0.75:
        partial["iou_achievement"] = 0.5
    elif best_iou >= 0.65:
        partial["iou_achievement"] = 0.3
    elif best_iou >= 0.55:
        partial["iou_achievement"] = 0.15

    best_lr, best_dice = None, None
    for a in state.action_history:
        if isinstance(a, TuneHyperparameters):
            best_lr = a.lr
        if isinstance(a, AdjustLossWeights):
            best_dice = a.dice_weight

    cfg = 0.0
    if best_lr is not None and 1e-4 <= best_lr <= 3e-3:
        cfg += 0.1
    if best_dice is not None and 0.55 <= best_dice <= 0.75:
        cfg += 0.1
    partial["config_quality"] = min(0.2, cfg)

    raw = partial["iou_achievement"] + (partial["config_quality"] if partial["iou_achievement"] < 1.0 else 0.0)
    score = _clamp(raw)

    return GraderResult(
        score=score,
        passed=score >= 0.84,
        reason=(
            f"Best IoU: {best_iou:.4f} "
            f"({'within' if achieved_within_limit else 'exceeded'} 5-step limit). "
            f"lr={best_lr}, dice_weight={best_dice}."
        ),
        partial_credit=partial,
    )


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, dict] = {
    "easy": {
        "id": "easy", "name": "CV Shape Error", "difficulty": "easy",
        "success_threshold": 0.99, "max_steps": 15,
        "description": "Fix a tensor reshape mismatch in a CNN. Flatten produces 512 features; 2304 are required.",
    },
    "medium": {
        "id": "medium", "name": "NLP Overfitting", "difficulty": "medium",
        "success_threshold": 0.8, "max_steps": 15,
        "description": "Resolve Transformer overfitting using dropout and sequence truncation.",
    },
    "hard": {
        "id": "hard", "name": "Lung Segmentation IoU", "difficulty": "hard",
        "success_threshold": 0.84, "max_steps": 15,
        "description": "Push lung segmentation IoU above 0.85 within 5 steps by tuning lr and Dice/CE weights.",
    },
}