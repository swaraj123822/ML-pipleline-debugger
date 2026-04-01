"""
env/simulator.py — Phase 2: The Fast-Simulation Engine

The "physics engine" that evaluates ML pipeline actions in <50ms with no GPU.
Operates as a deterministic state machine — same inputs always produce the
same outputs, making it fully reproducible for hackathon judging.

Public API:
    simulate_step(task_id, action, state) -> SimulationResult
    detect_crash(task_id, action, state)  -> CrashResult | None
    compute_loss_curve(task_id, action, state) -> list[EpochMetrics]
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import Literal

from env.models import (
    Action,
    AddAugmentation,
    AdjustLossWeights,
    EpochMetrics,
    FixReshape,
    TaskState,
    TuneHyperparameters,
)

# ---------------------------------------------------------------------------
# Internal result types (not exposed to agent)
# ---------------------------------------------------------------------------


@dataclass
class CrashResult:
    """Describes a simulated runtime failure."""
    crash_type: Literal["shape_mismatch", "oom", "nan_loss", "invalid_action"]
    error_trace: str
    is_recoverable: bool = True   # agent can try a different action


@dataclass
class SimulationResult:
    """Full output of simulate_step() — consumed by ml_debugger_env.py."""
    new_metrics: list[EpochMetrics]
    crash: CrashResult | None
    architecture_summary: str
    tensor_shapes: dict[str, list[int]]
    error_trace: str | None
    is_solved: bool
    reward_components: dict[str, float]   # broken-out signals for reward.py


# ---------------------------------------------------------------------------
# Task-specific ground truth configurations
# ---------------------------------------------------------------------------

# Easy Task — CV Shape Error
# The correct fix: layer="flatten", new_shape=[2304] (64 channels x 6 x 6 spatial)
_EASY_CORRECT_LAYER = "flatten"
_EASY_CORRECT_SHAPE = [2304]   # 64 * 6 * 6

# Medium Task — NLP Overfitting
# Solved when val_loss - train_loss < 0.15 AND at least one of dropout /
# truncate_sequence has been applied
_MEDIUM_GAP_THRESHOLD = 0.15
_MEDIUM_REQUIRED_STRATEGIES = {"dropout", "truncate_sequence"}

# Hard Task — Lung Segmentation
# Solved when IoU > 0.85 within 5 steps
_HARD_IOU_TARGET = 0.85
_HARD_STEP_LIMIT = 5

# Optimal hyperparameter ranges (used for heuristic scoring)
_OPTIMAL_LR_RANGE = (1e-4, 3e-3)
_OPTIMAL_DICE_RANGE = (0.55, 0.75)


# ---------------------------------------------------------------------------
# Deterministic pseudo-noise (no random — reproducible for judging)
# ---------------------------------------------------------------------------

def _det_noise(seed: str, scale: float = 0.01) -> float:
    """Tiny deterministic perturbation derived from a string seed."""
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    # Map to [-scale, +scale]
    return ((h % 10000) / 10000.0 - 0.5) * 2 * scale


# ---------------------------------------------------------------------------
# detect_crash()
# ---------------------------------------------------------------------------

def detect_crash(
    task_id: Literal["easy", "medium", "hard"],
    action: Action,
    state: TaskState,
) -> CrashResult | None:
    """
    Check whether the action triggers a simulated runtime error.
    Returns a CrashResult if the pipeline crashes, None otherwise.

    Crash conditions:
      Easy   — FixReshape with wrong shape triggers RuntimeError
      Medium — TuneHyperparameters with extreme lr triggers NaN loss
      Hard   — AdjustLossWeights with both weights near 0 triggers OOM proxy
      All    — Infinite-loop detection (same failed action repeated >= 3 times)
    """

    # --- Infinite loop detection (all tasks) ---
    if len(state.action_history) >= 3:
        last_three = state.action_history[-3:]
        if all(a.model_dump() == action.model_dump() for a in last_three):
            return CrashResult(
                crash_type="invalid_action",
                error_trace=(
                    "InfiniteLoopError: The agent has submitted the identical action "
                    "3 times in a row with no improvement. Execution halted."
                ),
                is_recoverable=True,
            )

    if task_id == "easy":
        return _detect_crash_easy(action)

    if task_id == "medium":
        return _detect_crash_medium(action)

    if task_id == "hard":
        return _detect_crash_hard(action, state)

    return None


def _detect_crash_easy(action: Action) -> CrashResult | None:
    """
    Easy task: only FixReshape can fix the shape mismatch.
    Any other action type leaves the crash unresolved (no new crash).
    A FixReshape with wrong dimensions triggers a fresh RuntimeError.
    """
    if not isinstance(action, FixReshape):
        return None

    if action.layer != _EASY_CORRECT_LAYER:
        return CrashResult(
            crash_type="shape_mismatch",
            error_trace=(
                f"RuntimeError: Tried to fix layer '{action.layer}' but the mismatch "
                f"originates at 'flatten'. mat1 and mat2 shapes cannot be multiplied "
                f"({action.new_shape} vs [512, 10])."
            ),
        )

    if action.new_shape != _EASY_CORRECT_SHAPE:
        product = math.prod(action.new_shape)
        return CrashResult(
            crash_type="shape_mismatch",
            error_trace=(
                f"RuntimeError: size mismatch for flatten -> fc1. "
                f"Expected flattened size 2304 (64x6x6) but got {product} "
                f"from shape {action.new_shape}. "
                f"Hint: Conv2d(3,64,3,padding=0) on 32x32 input -> 64x30x30 -> "
                f"after MaxPool2d(5) -> 64x6x6."
            ),
        )

    return None  # Correct fix — no crash


def _detect_crash_medium(action: Action) -> CrashResult | None:
    """Medium task: extreme learning rates produce NaN loss."""
    if isinstance(action, TuneHyperparameters):
        if action.lr > 0.5:
            return CrashResult(
                crash_type="nan_loss",
                error_trace=(
                    f"RuntimeError: Loss is NaN after epoch 1. "
                    f"Learning rate {action.lr} is too large for this Transformer. "
                    f"Gradients exploded. Try lr < 0.01."
                ),
            )
    return None


def _detect_crash_hard(action: Action, state: TaskState) -> CrashResult | None:
    """Hard task: badly imbalanced loss weights simulate near-zero gradient signal."""
    if isinstance(action, AdjustLossWeights):
        if action.dice_weight < 0.05 or action.ce_weight < 0.05:
            dominant = "Dice" if action.dice_weight > action.ce_weight else "CE"
            return CrashResult(
                crash_type="oom",
                error_trace=(
                    f"RuntimeError: Degenerate loss configuration. "
                    f"{dominant} loss dominates with weight > 0.95. "
                    f"Gradient signal collapsed — training diverged. "
                    f"Keep both weights above 0.05."
                ),
            )
    return None


# ---------------------------------------------------------------------------
# compute_loss_curve()
# ---------------------------------------------------------------------------

def compute_loss_curve(
    task_id: Literal["easy", "medium", "hard"],
    action: Action,
    state: TaskState,
    starting_epoch: int,
) -> list[EpochMetrics]:
    """
    Simulate the training trajectory produced by this action.
    Returns a list of EpochMetrics (one per simulated epoch).

    Uses mathematical heuristics — no actual training occurs.
    Results are deterministic given (task_id, action, step_number).
    """
    if task_id == "easy":
        return _curve_easy(action, state, starting_epoch)
    if task_id == "medium":
        return _curve_medium(action, state, starting_epoch)
    if task_id == "hard":
        return _curve_hard(action, state, starting_epoch)
    return []


def _curve_easy(action: Action, state: TaskState, epoch0: int) -> list[EpochMetrics]:
    """Easy task: shape mismatch means no epochs until fixed."""
    seed = f"easy-{epoch0}-{action.model_dump_json()}"

    if (
        isinstance(action, FixReshape)
        and action.layer == _EASY_CORRECT_LAYER
        and action.new_shape == _EASY_CORRECT_SHAPE
    ):
        base_train, base_val = 2.31, 2.35
        metrics = []
        for i in range(3):
            drop = (i + 1) * 0.35 + _det_noise(seed + str(i), 0.02)
            metrics.append(EpochMetrics(
                epoch=epoch0 + i,
                train_loss=round(max(0.1, base_train - drop), 4),
                val_loss=round(max(0.15, base_val - drop * 0.9), 4),
                accuracy=round(min(0.98, 0.1 + (i + 1) * 0.28 + _det_noise(seed, 0.01)), 4),
            ))
        return metrics

    return []  # Shape still broken — no epochs run


def _curve_medium(action: Action, state: TaskState, epoch0: int) -> list[EpochMetrics]:
    """Medium task — overfitting scenario with accumulating augmentation strategies."""
    seed = f"medium-{epoch0}-{action.model_dump_json()}"

    applied: set[str] = set()
    for past_action in state.action_history:
        if isinstance(past_action, AddAugmentation):
            applied.add(past_action.strategy)
    if isinstance(action, AddAugmentation):
        applied.add(action.strategy)

    current_lr = 5e-4
    current_epochs = 5
    for past in reversed(state.action_history):
        if isinstance(past, TuneHyperparameters):
            current_lr = past.lr
            current_epochs = past.epochs
            break
    if isinstance(action, TuneHyperparameters):
        current_lr = action.lr
        current_epochs = action.epochs

    gap_reduction = 0.0
    if "dropout" in applied:
        gap_reduction += 0.12
    if "truncate_sequence" in applied:
        gap_reduction += 0.10
    if "weight_decay" in applied:
        gap_reduction += 0.05
    if current_lr > 0.01:
        gap_reduction -= 0.08

    num_epochs = min(current_epochs, 5)
    metrics = []
    base_train, base_val = 1.80, 2.40

    for i in range(num_epochs):
        noise = _det_noise(seed + str(i), 0.015)
        progress = (i + 1) / num_epochs
        train_loss = round(max(0.05, base_train - progress * 0.6 + noise), 4)
        val_loss = round(
            max(train_loss + 0.02, base_val - progress * (0.4 + gap_reduction) + noise), 4
        )
        metrics.append(EpochMetrics(
            epoch=epoch0 + i,
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=round(min(0.95, 0.45 + progress * 0.3 + _det_noise(seed, 0.01)), 4),
        ))

    return metrics


def _curve_hard(action: Action, state: TaskState, epoch0: int) -> list[EpochMetrics]:
    """Hard task — lung segmentation IoU trajectory."""
    seed = f"hard-{epoch0}-{action.model_dump_json()}"

    best_lr = 1e-3
    best_dice = 0.5
    for past in state.action_history:
        if isinstance(past, TuneHyperparameters):
            best_lr = past.lr
        if isinstance(past, AdjustLossWeights):
            best_dice = past.dice_weight
    if isinstance(action, TuneHyperparameters):
        best_lr = action.lr
    if isinstance(action, AdjustLossWeights):
        best_dice = action.dice_weight

    lr_score = _lr_optimality(best_lr)
    dice_score = _dice_optimality(best_dice)
    config_score = lr_score * 0.45 + dice_score * 0.55

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
        iou = round(
            min(iou_ceiling, base_iou + progress * (iou_ceiling - base_iou) * 0.6 + noise), 4
        )
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
    action: Action,
    state: TaskState,
) -> SimulationResult:
    """
    Core simulation entry point. Called by MLDebuggerEnv.step().

    1. detect_crash() — if crash, return immediately with penalty signals.
    2. compute_loss_curve() — generate epoch metrics.
    3. Evaluate whether the task is solved.
    4. Build reward_components dict for reward.py to consume.
    """
    starting_epoch = (
        state.last_observation.last_metrics.epoch + 1
        if state.last_observation and state.last_observation.last_metrics
        else 0
    )

    crash = detect_crash(task_id, action, state)

    if crash:
        arch, shapes = _get_architecture(task_id, action, state, crashed=True)
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
    reward_components = _build_reward_components(task_id, action, state, new_metrics, is_solved)
    arch, shapes = _get_architecture(task_id, action, state, crashed=False)

    return SimulationResult(
        new_metrics=new_metrics,
        crash=None,
        architecture_summary=arch,
        tensor_shapes=shapes,
        error_trace=None,
        is_solved=is_solved,
        reward_components=reward_components,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_solved(
    task_id: str,
    action: Action,
    state: TaskState,
    new_metrics: list[EpochMetrics],
) -> bool:
    if task_id == "easy":
        return (
            isinstance(action, FixReshape)
            and action.layer == _EASY_CORRECT_LAYER
            and action.new_shape == _EASY_CORRECT_SHAPE
            and len(new_metrics) > 0
        )

    if task_id == "medium":
        if not new_metrics:
            return False
        last = new_metrics[-1]
        gap = last.val_loss - last.train_loss
        applied: set[str] = set()
        for past in state.action_history:
            if isinstance(past, AddAugmentation):
                applied.add(past.strategy)
        if isinstance(action, AddAugmentation):
            applied.add(action.strategy)
        return bool(applied & _MEDIUM_REQUIRED_STRATEGIES) and gap < _MEDIUM_GAP_THRESHOLD

    if task_id == "hard":
        if not new_metrics:
            return False
        last_iou = new_metrics[-1].iou or 0.0
        return last_iou >= _HARD_IOU_TARGET and state.current_step < _HARD_STEP_LIMIT

    return False


def _build_reward_components(
    task_id: str,
    action: Action,
    state: TaskState,
    new_metrics: list[EpochMetrics],
    is_solved: bool,
) -> dict[str, float]:
    epoch_bonus = len(new_metrics) * 0.1

    loss_improvement = 0.0
    if new_metrics and state.last_observation and state.last_observation.last_metrics:
        prev_val = state.last_observation.last_metrics.val_loss
        new_val = new_metrics[-1].val_loss
        drop = prev_val - new_val
        if drop > 0:
            loss_improvement = (drop / 0.05) * 1.0

    return {
        "epoch_bonus": round(epoch_bonus, 4),
        "loss_improvement": round(loss_improvement, 4),
        "crash_penalty": 0.0,
        "solve_bonus": 2.0 if is_solved else 0.0,
    }


def _get_architecture(
    task_id: str,
    action: Action,
    state: TaskState,
    crashed: bool,
) -> tuple[str, dict[str, list[int]]]:
    """Return architecture summary and tensor shapes for the Observation."""

    if task_id == "easy":
        if (
            isinstance(action, FixReshape)
            and action.layer == _EASY_CORRECT_LAYER
            and action.new_shape == _EASY_CORRECT_SHAPE
            and not crashed
        ):
            return (
                "CNN (FIXED): Conv2d(3,64,3,pad=0) -> ReLU -> MaxPool2d(5) -> "
                "Flatten([2304]) -> Linear(2304,512) -> ReLU -> Linear(512,10)",
                {
                    "input": [1, 3, 32, 32],
                    "conv1_out": [1, 64, 30, 30],
                    "pool_out": [1, 64, 6, 6],
                    "flatten_out": [1, 2304],
                    "fc1_out": [1, 512],
                    "output": [1, 10],
                },
            )
        return (
            "CNN (BROKEN): Conv2d(3,64,3,pad=0) -> ReLU -> MaxPool2d(5) -> "
            "Flatten([512]) <- WRONG -> Linear(512,10)",
            {
                "input": [1, 3, 32, 32],
                "conv1_out": [1, 64, 30, 30],
                "pool_out": [1, 64, 6, 6],
                "flatten_out": [1, 512],
                "fc1_in_expected": [1, 2304],
            },
        )

    if task_id == "medium":
        return (
            "Transformer: Embedding(30522,256) -> 4xTransformerEncoderLayer(256,8heads) "
            "-> Linear(256,2) [binary classifier, imbalanced data]",
            {
                "input_ids": [16, 512],
                "embedding_out": [16, 512, 256],
                "encoder_out": [16, 512, 256],
                "cls_token": [16, 256],
                "logits": [16, 2],
            },
        )

    # hard
    return (
        "SegNet: ResNet50-Encoder -> AdaptiveAttention(512) -> "
        "LightweightDecoder(256,128,64) -> Sigmoid [lung segmentation, 5-class]",
        {
            "input": [2, 3, 512, 512],
            "encoder_out": [2, 512, 16, 16],
            "attention_out": [2, 512, 16, 16],
            "decoder_out": [2, 5, 512, 512],
            "prediction": [2, 5, 512, 512],
        },
    )

