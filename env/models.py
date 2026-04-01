"""
env/models.py — Strictly typed Pydantic v2 data models.

Defines the complete language an LLM agent uses to interact with the
ML Pipeline Debugger environment:
  - Action space:   Discriminated union of 4 concrete action types
  - Observation:    JSON-serializable snapshot of pipeline state
  - Reward:         Dense signal with reason and cumulative tracking
  - StepResult:     Return type of step()
  - TaskState:      Internal env state serialized by state()
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Action Space — Discriminated Union
# ---------------------------------------------------------------------------


class TuneHyperparameters(BaseModel):
    """
    Adjust training hyperparameters.
    Valid for all tasks. Primary lever for fixing overfitting and loss curves.
    """

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


class FixReshape(BaseModel):
    """
    Correct a tensor reshape operation in the pipeline.
    Primary action for the Easy task (CV shape error).
    """

    action_type: Literal["fix_reshape"] = "fix_reshape"
    layer: str = Field(..., description="Name of the layer to fix (e.g. 'flatten', 'fc1').")
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


class AddAugmentation(BaseModel):
    """
    Apply a regularisation or data-augmentation strategy.
    Primary action for the Medium task (NLP overfitting).
    """

    action_type: Literal["add_augmentation"] = "add_augmentation"
    strategy: Literal[
        "dropout",
        "weight_decay",
        "truncate_sequence",
        "horizontal_flip",
        "mixup",
    ] = Field(..., description="Augmentation strategy to apply.")


class AdjustLossWeights(BaseModel):
    """
    Rebalance composite loss function weights (Dice vs Cross-Entropy).
    Required for the Hard task (lung segmentation IoU > 0.85).
    """

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


# The single Action type the environment accepts — discriminated on action_type.
Action = Annotated[
    Union[TuneHyperparameters, FixReshape, AddAugmentation, AdjustLossWeights],
    Field(discriminator="action_type"),
]


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------


class EpochMetrics(BaseModel):
    """Per-epoch training metrics. Appended to metrics_history each step."""

    epoch: int = Field(..., ge=0, description="Epoch index (0-based).")
    train_loss: float = Field(..., ge=0.0, description="Training loss.")
    val_loss: float = Field(..., ge=0.0, description="Validation loss.")
    accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Classification accuracy (optional)."
    )
    iou: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Intersection over Union (Hard task only)."
    )


class Observation(BaseModel):
    """
    Full JSON-serializable snapshot of the pipeline state.
    This is what the LLM agent receives at every step.
    """

    task_id: Literal["easy", "medium", "hard"] = Field(
        ..., description="Which debugging task is active."
    )
    architecture_summary: str = Field(
        ..., description="Human-readable description of the model architecture."
    )
    tensor_shapes: dict[str, list[int]] = Field(
        ..., description="Layer-name → shape mapping for all key tensors."
    )
    error_trace: str | None = Field(
        default=None,
        description="Last RuntimeError trace, or None if the forward pass succeeded.",
    )
    metrics_history: list[EpochMetrics] = Field(
        default_factory=list,
        description="Full training history so far, one entry per simulated epoch.",
    )
    step_number: int = Field(..., ge=0, description="Current step index (0-based).")
    max_steps: int = Field(
        default=15, gt=0, description="Maximum steps allowed per episode."
    )

    @property
    def last_metrics(self) -> EpochMetrics | None:
        """Convenience accessor for the most recent epoch metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    @property
    def steps_remaining(self) -> int:
        return self.max_steps - self.step_number


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class Reward(BaseModel):
    """
    Dense reward signal for a single step.

    Reward schedule (from execution plan):
      +0.1  per successful (non-crashing) simulated epoch
      +1.0  for every 0.05 drop in validation loss
      -0.5  for actions causing a crash, OOM, or NaN loss
      -0.3  for infinite-loop detection (repeating same failed action)
    """

    value: float = Field(..., description="Reward for this step.")
    reason: str = Field(..., description="Human-readable explanation of reward signal.")
    cumulative: float = Field(..., description="Total reward accumulated this episode.")

    @field_validator("value")
    @classmethod
    def clamp_value(cls, v: float) -> float:
        # Hard clamp — no single step should swing beyond these bounds.
        return max(-1.0, min(v, 2.0))


# ---------------------------------------------------------------------------
# StepResult — return type of step()
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """Exact return value of MLDebuggerEnv.step(action)."""

    observation: Observation
    reward: Reward
    done: bool = Field(..., description="True if the episode has ended.")
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary diagnostics (grader score, crash type, etc.).",
    )


# ---------------------------------------------------------------------------
# TaskState — internal state serialized by state()
# ---------------------------------------------------------------------------


class TaskState(BaseModel):
    """
    Full internal state of a running episode.
    Returned by MLDebuggerEnv.state() and used for checkpointing.
    Not exposed to the agent directly.
    """

    task_id: Literal["easy", "medium", "hard"]
    current_step: int = Field(default=0, ge=0)
    is_solved: bool = Field(default=False)
    cumulative_reward: float = Field(default=0.0)
    action_history: list[
        Union[TuneHyperparameters, FixReshape, AddAugmentation, AdjustLossWeights]
    ] = Field(default_factory=list)
    last_observation: Observation | None = Field(default=None)

    @property
    def is_terminal(self) -> bool:
        return self.is_solved or self.current_step >= 15
    
print('''models.py loaded successfully
      This module defines the core data models for the ML Pipeline Debugger environment, including:
      - Action: the set of valid actions the agent can take, as a discriminated union of 4 types.
      - Observation: the structured state information the agent receives at each step.
      - Reward: the dense reward signal with value, reason, and cumulative tracking.
      ''')    
    