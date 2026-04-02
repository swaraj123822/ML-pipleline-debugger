"""
models.py — Shared type-safe contracts between client and server.

Inherits from openenv-core base types:
  - Action      → openenv.core.env_server.types.Action
  - Observation → openenv.core.env_server.types.Observation
  - State       → openenv.core.env_server.types.State

These models are imported by BOTH client.py and server/environment.py.
Never import from server/ in this file — it must stay dependency-free
so the client can use it without pulling in FastAPI/uvicorn.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from openenv.core.env_server.types import Action as _BaseAction
from openenv.core.env_server.types import Observation as _BaseObservation
from openenv.core.env_server.types import State as _BaseState
from pydantic import Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Action Space — Discriminated Union of 4 concrete action types
# ---------------------------------------------------------------------------


class TuneHyperparameters(_BaseAction):
    """Adjust training hyperparameters. Valid for all tasks."""
    action_type: Literal["tune_hyperparameters"] = "tune_hyperparameters"
    lr: float = Field(..., description="Learning rate. Must be in (0.0, 1.0).")
    batch_size: int = Field(..., description="Batch size. Must be a power of 2.")
    epochs: int = Field(..., description="Number of training epochs. Range [1, 50].")

    @field_validator("lr")
    @classmethod
    def lr_in_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"lr must be in (0.0, 1.0), got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def batch_size_power_of_two(cls, v: int) -> int:
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError(f"batch_size must be a positive power of 2, got {v}")
        return v

    @field_validator("epochs")
    @classmethod
    def epochs_in_range(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError(f"epochs must be in [1, 50], got {v}")
        return v


class FixReshape(_BaseAction):
    """Correct a tensor reshape operation. Primary action for the Easy task."""
    action_type: Literal["fix_reshape"] = "fix_reshape"
    layer: str = Field(..., description="Name of the layer to fix (e.g. 'flatten').")
    new_shape: list[int] = Field(
        ..., description="Target tensor shape as a list of positive ints. Max 4 dims."
    )

    @field_validator("layer")
    @classmethod
    def layer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("layer name must not be empty")
        return v.strip()

    @field_validator("new_shape")
    @classmethod
    def shape_is_valid(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("new_shape must not be empty")
        if len(v) > 4:
            raise ValueError(f"new_shape may have at most 4 dimensions, got {len(v)}")
        if any(d <= 0 for d in v):
            raise ValueError(f"All shape dimensions must be > 0, got {v}")
        return v


class AddAugmentation(_BaseAction):
    """Apply a regularisation or data-augmentation strategy. Primary for Medium task."""
    action_type: Literal["add_augmentation"] = "add_augmentation"
    strategy: Literal[
        "dropout",
        "weight_decay",
        "truncate_sequence",
        "horizontal_flip",
        "mixup",
    ] = Field(..., description="Augmentation strategy to apply.")


class AdjustLossWeights(_BaseAction):
    """Rebalance Dice vs Cross-Entropy loss weights. Required for Hard task."""
    action_type: Literal["adjust_loss_weights"] = "adjust_loss_weights"
    dice_weight: float = Field(..., description="Weight for Dice loss. Range [0.0, 1.0].")
    ce_weight: float = Field(..., description="Weight for Cross-Entropy loss. Range [0.0, 1.0].")

    @field_validator("dice_weight", "ce_weight")
    @classmethod
    def weight_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Loss weight must be in [0.0, 1.0], got {v}")
        return round(v, 6)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "AdjustLossWeights":
        total = self.dice_weight + self.ce_weight
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(
                f"dice_weight + ce_weight must equal 1.0, got {total:.6f}"
            )
        return self


# Single Action type — discriminated on action_type field
Action = Annotated[
    Union[TuneHyperparameters, FixReshape, AddAugmentation, AdjustLossWeights],
    Field(discriminator="action_type"),
]


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------


class EpochMetrics(_BaseObservation):
    """Per-epoch training metrics. Appended to metrics_history each step."""
    epoch: int = Field(..., ge=0)
    train_loss: float = Field(..., ge=0.0)
    val_loss: float = Field(..., ge=0.0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    iou: float | None = Field(default=None, ge=0.0, le=1.0)


class MLDebuggerObservation(_BaseObservation):
    """
    Full JSON-serializable snapshot of the pipeline state.
    Received by the agent at every step via the WebSocket client.
    """
    task_id: Literal["easy", "medium", "hard"]
    architecture_summary: str
    tensor_shapes: dict[str, list[int]]
    error_trace: str | None = Field(default=None)
    metrics_history: list[EpochMetrics] = Field(default_factory=list)
    step_number: int = Field(..., ge=0)
    max_steps: int = Field(default=15, gt=0)

    @property
    def last_metrics(self) -> EpochMetrics | None:
        return self.metrics_history[-1] if self.metrics_history else None

    @property
    def steps_remaining(self) -> int:
        return self.max_steps - self.step_number


# ---------------------------------------------------------------------------
# State — internal episode state serialized by state()
# ---------------------------------------------------------------------------


class MLDebuggerState(_BaseState):
    """
    Full internal episode state. Returned by the server's state() call.
    Not exposed to the agent directly — used for checkpointing/debugging.
    """
    task_id: Literal["easy", "medium", "hard"]
    current_step: int = Field(default=0, ge=0)
    is_solved: bool = Field(default=False)
    cumulative_reward: float = Field(default=0.0)
    action_history: list[
        Union[TuneHyperparameters, FixReshape, AddAugmentation, AdjustLossWeights]
    ] = Field(default_factory=list)
    last_observation: MLDebuggerObservation | None = Field(default=None)

    @property
    def is_terminal(self) -> bool:
        return self.is_solved or self.current_step >= 15


# ---------------------------------------------------------------------------
# StepResult — payload returned to client after each step()
# ---------------------------------------------------------------------------


class StepResult(_BaseObservation):
    """What the WebSocket client receives after calling step()."""
    observation: MLDebuggerObservation
    reward: float
    reward_reason: str
    cumulative_reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Re-exports for convenience
# ---------------------------------------------------------------------------

__all__ = [
    "Action",
    "AddAugmentation",
    "AdjustLossWeights",
    "EpochMetrics",
    "FixReshape",
    "MLDebuggerObservation",
    "MLDebuggerState",
    "StepResult",
    "TuneHyperparameters",
]