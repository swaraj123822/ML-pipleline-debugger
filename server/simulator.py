"""
server/simulator.py — Fast-Simulation Engine

Evaluates ML pipeline actions in <50ms with no GPU.
Deterministic state machine — same inputs always produce same outputs.

Public API:
    simulate_step(task_id, action, state) -> SimulationResult
    detect_crash(task_id, action, state)  -> CrashResult | None
    compute_loss_curve(...)               -> list[EpochMetrics]
"""

from __future__ import annotations

import hashlib
import math
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

# ---------------------------------------------------------------------------
# Ground truth config
# ---------------------------------------------------------------------------

_EASY_CORRECT_LAYER = "flatten"
_EASY_CORRECT_SHAPE = [2304]          # 64 * 6 * 6

_MEDIUM_GAP_THRESHOLD = 0.15
_MEDIUM_REQUIRED_STRATEGIES = {"dropout", "truncate_sequence"}

_HARD_IOU_TARGET = 0.85
_HARD_STEP_LIMIT = 5

_OPTIMAL_LR_RANGE = (1e-4, 3e-3)
_OPTIMAL_DICE_RANGE = (0.55, 0.75)


# ---------------------------------------------------------------------------
# Internal result types
# ---------------------------------------------------------------------------


@dataclass
class CrashResult:
    crash_type: Literal["shape_mismatch", "oom", "nan_loss", "invalid_action"]
    error_trace: str
    is_recoverable: bool = True


@dataclass
class SimulationResult:
    new_metrics: list[EpochMetrics]
    crash: CrashResult | None
    architecture_summary: str
    tensor_shapes: dict[str, list[int]]
    error_trace: str | None
    is_solved: bool
    reward_components: dict[str, float]


# ---------------------------------------------------------------------------
# Deterministic noise
# ---------------------------------------------------------------------------

def _det_noise(seed: str, scale: float = 0.01) -> float:
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    return ((h % 10000) / 10000.0 - 0.5) * 2 * scale


# ---------------------------------------------------------------------------
# detect_crash()
# ---------------------------------------------------------------------------

def detect_crash(
    task_id: Literal["easy", "medium", "hard"],
    action: object,
    state: MLDebuggerState,
) -> CrashResult | None:
    # Infinite loop detection — all tasks
    if len(state.action_history) >= 3:
        last_three = state.action_history[-3:]
        if all(a.model_dump() == action.model_dump() for a in last_three):
            return CrashResult(
                crash_type="invalid_action",
                error_trace=(
                    "InfiniteLoopError: Identical action submitted 3 times in a row "
                    "with no improvement. Try a different approach."
                ),
            )

    if task_id == "easy":
        return _detect_crash_easy(action)
    if task_id == "medium":
        return _detect_crash_medium(action)
    if task_id == "hard":
        return _detect_crash_hard(action)
    return None


def _detect_crash_easy(action: object) -> CrashResult | None:
    if not isinstance(action, FixReshape):
        return None
    if action.layer != _EASY_CORRECT_LAYER:
        return CrashResult(
            crash_type="shape_mismatch",
            error_trace=(
                f"RuntimeError: Tried to fix layer '{action.layer}' but the mismatch "
                f"originates at 'flatten'. mat1 and mat2 shapes cannot be multiplied "
                f"({action.new_shape} vs [2304, 512])."
            ),
        )
    if action.new_shape != _EASY_CORRECT_SHAPE:
        product = math.prod(action.new_shape)
        return CrashResult(
            crash_type="shape_mismatch",
            error_trace=(
                f"RuntimeError: size mismatch for flatten -> fc1. "
                f"Expected 2304 (64x6x6) but got {product} from shape {action.new_shape}. "
                f"Hint: Conv2d(3,64,3,pad=0) on 32x32 -> 64x30x30 -> MaxPool2d(5) -> 64x6x6."
            ),
        )
    return None


def _detect_crash_medium(action: object) -> CrashResult | None:
    if isinstance(action, TuneHyperparameters) and action.lr > 0.5:
        return CrashResult(
            crash_type="nan_loss",
            error_trace=(
                f"RuntimeError: Loss is NaN after epoch 1. "
                f"lr={action.lr} is too large — gradients exploded. Try lr < 0.01."
            ),
        )
    return None


def _detect_crash_hard(action: object) -> CrashResult | None:
    if isinstance(action, AdjustLossWeights):
        if action.dice_weight < 0.05 or action.ce_weight < 0.05:
            dominant = "Dice" if action.dice_weight > action.ce_weight else "CE"
            return CrashResult(
                crash_type="oom",
                error_trace=(
                    f"RuntimeError: Degenerate loss — {dominant} weight > 0.95. "
                    f"Gradient signal collapsed. Keep both weights above 0.05."
                ),
            )
    return None


# ---------------------------------------------------------------------------
# compute_loss_curve()
# ---------------------------------------------------------------------------

def compute_loss_curve(
    task_id: Literal["easy", "medium", "hard"],
    action: object,
    state: MLDebuggerState,
    starting_epoch: int,
) -> list[EpochMetrics]:
    if task_id == "easy":
        return _curve_easy(action, state, starting_epoch)
    if task_id == "medium":
        return _curve_medium(action, state, starting_epoch)
    if task_id == "hard":
        return _curve_hard(action, state, starting_epoch)
    return []


def _curve_easy(action: object, state: MLDebuggerState, epoch0: int) -> list[EpochMetrics]:
    seed = f"easy-{epoch0}-{action.model_dump_json()}"
    if (
        isinstance(action, FixReshape)
        and action.layer == _EASY_CORRECT_LAYER
        and action.new_shape == _EASY_CORRECT_SHAPE
    ):
        metrics = []
        for i in range(3):
            drop = (i + 1) * 0.35 + _det_noise(seed + str(i), 0.02)
            metrics.append(EpochMetrics(
                epoch=epoch0 + i,
                train_loss=round(max(0.1, 2.31 - drop), 4),
                val_loss=round(max(0.15, 2.35 - drop * 0.9), 4),
                accuracy=round(min(0.98, 0.1 + (i + 1) * 0.28 + _det_noise(seed, 0.01)), 4),
            ))
        return metrics
    return []


def _curve_medium(action: object, state: MLDebuggerState, epoch0: int) -> list[EpochMetrics]:
    seed = f"medium-{epoch0}-{action.model_dump_json()}"

    applied: set[str] = set()
    for past in state.action_history:
        if isinstance(past, AddAugmentation):
            applied.add(past.strategy)
    if isinstance(action, AddAugmentation):
        applied.add(action.strategy)

    current_lr, current_epochs = 5e-4, 5
    for past in reversed(state.action_history):
        if isinstance(past, TuneHyperparameters):
            current_lr, current_epochs = past.lr, past.epochs
            break
    if isinstance(action, TuneHyperparameters):
        current_lr, current_epochs = action.lr, action.epochs

    gap_reduction = 0.0
    if "dropout" in applied:
        gap_reduction += 0.12
    if "truncate_sequence" in applied:
        gap_reduction += 0.10
    if "weight_decay" in applied:
        gap_reduction += 0.05
    if current_lr > 0.01:
        gap_reduction -= 0.08

    metrics = []
    for i in range(min(current_epochs, 5)):
        noise = _det_noise(seed + str(i), 0.015)
        progress = (i + 1) / min(current_epochs, 5)
        train_loss = round(max(0.05, 1.80 - progress * 0.6 + noise), 4)
        val_loss = round(
            max(train_loss + 0.02, 2.40 - progress * (0.4 + gap_reduction) + noise), 4
        )
        metrics.append(EpochMetrics(
            epoch=epoch0 + i,
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=round(min(0.95, 0.45 + progress * 0.3 + _det_noise(seed, 0.01)), 4),
        ))
    return metrics


def _curve_hard(action: object, state: MLDebuggerState, epoch0: int) -> list[EpochMetrics]:
    seed = f"hard-{epoch0}-{action.model_dump_json()}"

    best_lr, best_dice = 1e-3, 0.5
    for past in state.action_history:
        if isinstance(past, TuneHyperparameters):
            best_lr = past.lr
        if isinstance(past, AdjustLossWeights):
            best_dice = past.dice_weight
    if isinstance(action, TuneHyperparameters):
        best_lr = action.lr
    if isinstance(action, AdjustLossWeights):
        best_dice = action.dice_weight

    config_score = _lr_optimality(best_lr) * 0.45 + _dice_optimality(best_dice) * 0.55
    iou_ceiling = 0.52 + config_score * 0.45

    base_iou = 0.52
    if (
        state.last_observation
        and state.last_observation.last_metrics
        and state.last_observation.last_metrics.iou is not None
    ):
        base_iou = max(0.52, state.last_observation.last_metrics.iou)

    metrics = []
    for i in range(3):
        noise = _det_noise(seed + str(i), 0.008)
        progress = (i + 1) / 3
        iou = round(min(iou_ceiling, base_iou + progress * (iou_ceiling - base_iou) * 0.6 + noise), 4)
        train_loss = round(max(0.08, 0.85 - progress * 0.15 * config_score + noise), 4)
        val_loss = round(train_loss + 0.04 + _det_noise(seed, 0.01), 4)
        metrics.append(EpochMetrics(
            epoch=epoch0 + i,
            train_loss=train_loss,
            val_loss=val_loss,
            iou=iou,
        ))
    return metrics


def _lr_optimality(lr: float) -> float:
    lo, hi = _OPTIMAL_LR_RANGE
    if lo <= lr <= hi:
        return 1.0
    if lr < lo:
        return max(0.0, 1.0 - (lo - lr) / lo * 5)
    return max(0.0, 1.0 - (lr - hi) / hi * 3)


def _dice_optimality(dice: float) -> float:
    lo, hi = _OPTIMAL_DICE_RANGE
    if lo <= dice <= hi:
        return 1.0
    if dice < lo:
        return max(0.0, 1.0 - (lo - dice) / lo * 4)
    return max(0.0, 1.0 - (dice - hi) / (1.0 - hi) * 4)


# ---------------------------------------------------------------------------
# simulate_step() — main entry point
# ---------------------------------------------------------------------------

def simulate_step(
    task_id: Literal["easy", "medium", "hard"],
    action: object,
    state: MLDebuggerState,
) -> SimulationResult:
    starting_epoch = (
        state.last_observation.last_metrics.epoch + 1
        if state.last_observation and state.last_observation.last_metrics
        else 0
    )

    crash = detect_crash(task_id, action, state)
    if crash:
        arch, shapes = _get_architecture(task_id, action, crashed=True)
        return SimulationResult(
            new_metrics=[],
            crash=crash,
            architecture_summary=arch,
            tensor_shapes=shapes,
            error_trace=crash.error_trace,
            is_solved=False,
            reward_components={
                "epoch_bonus": 0.0,
                "loss_improvement": 0.0,
                "crash_penalty": -0.5 if crash.crash_type != "invalid_action" else -0.3,
                "solve_bonus": 0.0,
            },
        )

    new_metrics = compute_loss_curve(task_id, action, state, starting_epoch)
    is_solved = _check_solved(task_id, action, state, new_metrics)

    loss_improvement = 0.0
    if new_metrics and state.last_observation and state.last_observation.last_metrics:
        drop = state.last_observation.last_metrics.val_loss - new_metrics[-1].val_loss
        if drop > 0:
            loss_improvement = round((drop / 0.05) * 1.0, 4)

    arch, shapes = _get_architecture(task_id, action, crashed=False)

    return SimulationResult(
        new_metrics=new_metrics,
        crash=None,
        architecture_summary=arch,
        tensor_shapes=shapes,
        error_trace=None,
        is_solved=is_solved,
        reward_components={
            "epoch_bonus": round(len(new_metrics) * 0.1, 4),
            "loss_improvement": loss_improvement,
            "crash_penalty": 0.0,
            "solve_bonus": 2.0 if is_solved else 0.0,
        },
    )


def _check_solved(task_id: str, action: object, state: MLDebuggerState, metrics: list) -> bool:
    if task_id == "easy":
        return (
            isinstance(action, FixReshape)
            and action.layer == _EASY_CORRECT_LAYER
            and action.new_shape == _EASY_CORRECT_SHAPE
            and len(metrics) > 0
        )
    if task_id == "medium":
        if not metrics:
            return False
        gap = metrics[-1].val_loss - metrics[-1].train_loss
        applied: set[str] = set()
        for past in state.action_history:
            if isinstance(past, AddAugmentation):
                applied.add(past.strategy)
        if isinstance(action, AddAugmentation):
            applied.add(action.strategy)
        return bool(applied & _MEDIUM_REQUIRED_STRATEGIES) and gap < _MEDIUM_GAP_THRESHOLD
    if task_id == "hard":
        if not metrics:
            return False
        return (metrics[-1].iou or 0.0) >= _HARD_IOU_TARGET and state.current_step < _HARD_STEP_LIMIT
    return False


def _get_architecture(task_id: str, action: object, crashed: bool) -> tuple[str, dict]:
    if task_id == "easy":
        if isinstance(action, FixReshape) and action.layer == "flatten" and action.new_shape == [2304] and not crashed:
            return (
                "CNN (FIXED): Conv2d(3,64,3,pad=0)->ReLU->MaxPool2d(5)->Flatten([2304])->Linear(2304,512)->ReLU->Linear(512,10)",
                {"input": [1,3,32,32], "conv1_out": [1,64,30,30], "pool_out": [1,64,6,6], "flatten_out": [1,2304], "fc1_out": [1,512], "output": [1,10]},
            )
        return (
            "CNN (BROKEN): Conv2d(3,64,3,pad=0)->ReLU->MaxPool2d(5)->Flatten([512])<-WRONG->Linear(512,10)",
            {"input": [1,3,32,32], "conv1_out": [1,64,30,30], "pool_out": [1,64,6,6], "flatten_out": [1,512], "fc1_in_expected": [1,2304]},
        )
    if task_id == "medium":
        return (
            "Transformer: Embedding(30522,256)->4xTransformerEncoderLayer(256,8heads)->Linear(256,2)",
            {"input_ids": [16,512], "embedding_out": [16,512,256], "encoder_out": [16,512,256], "cls_token": [16,256], "logits": [16,2]},
        )
    return (
        "SegNet: ResNet50->AdaptiveAttention(512)->LightweightDecoder(256,128,64)->Sigmoid [5-class lung seg]",
        {"input": [2,3,512,512], "encoder_out": [2,512,16,16], "attention_out": [2,512,16,16], "decoder_out": [2,5,512,512]},
    )