"""
server/environment.py — MLDebuggerEnvironment

Extends openenv.core.env_server.interfaces.Environment.
Contains all game logic — wraps simulator, tasks, and reward.
Instantiated per WebSocket session by app.py.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from models import (
    AdjustLossWeights,
    AddAugmentation,
    EpochMetrics,
    FixReshape,
    MLDebuggerObservation,
    MLDebuggerState,
    StepResult,
    TuneHyperparameters,
    ChangeOptimizer,
    ToggleLayerFreeze
)
from server.reward import compute_reward
from server.simulator import simulate_step
from server.tasks import TASK_REGISTRY, get_initial_observation, grade

MAX_STEPS = 15


class MLDebuggerEnvironment(Environment):
    """
    OpenEnv-compliant ML Pipeline Debugger environment.

    One instance is created per WebSocket session (see app.py).
    The task_id is passed on reset() so a single server can serve all 3 tasks.
    """

    def __init__(self) -> None:
        self._state: MLDebuggerState | None = None
        self._task_id: Literal["easy", "medium", "hard"] = "easy"

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: Literal["easy", "medium", "hard"] = "easy") -> MLDebuggerObservation:
        """
        Start a new episode for the given task.
        Returns the initial observation.
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose: easy | medium | hard")

        self._task_id = task_id
        initial_obs = get_initial_observation(task_id)
        self._state = MLDebuggerState(
            task_id=task_id,
            current_step=0,
            is_solved=False,
            cumulative_reward=0.0,
            action_history=[],
            last_observation=initial_obs,
        )
        return initial_obs

    def step(self, action: Any) -> StepResult:
        """
        Apply a validated action. Returns StepResult.
        Raises RuntimeError if reset() hasn't been called.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        # Already terminal — return last obs with zero reward
        if self._state.is_terminal:
            return StepResult(
                observation=self._state.last_observation,
                reward=0.0,
                reward_reason="Episode already terminated. Call reset() to start a new episode.",
                cumulative_reward=self._state.cumulative_reward,
                done=True,
                info={"warning": "step() called after episode termination"},
            )

        # Deserialize action if it arrived as a dict (from WebSocket JSON)
        if isinstance(action, dict):
            action = _parse_action(action)

        # Run simulation
        sim = simulate_step(self._task_id, action, self._state)

        # Compute reward
        reward_value, reward_reason = compute_reward(sim, self._state.cumulative_reward)
        new_cumulative = round(self._state.cumulative_reward + reward_value, 6)

        # Build new observation
        new_metrics_history = list(self._state.last_observation.metrics_history) + sim.new_metrics
        new_obs = MLDebuggerObservation(
            task_id=self._task_id,
            architecture_summary=sim.architecture_summary,
            tensor_shapes=sim.tensor_shapes,
            error_trace=sim.error_trace,
            metrics_history=new_metrics_history,
            step_number=self._state.current_step + 1,
            max_steps=MAX_STEPS,
        )

        # Update state
        self._state = MLDebuggerState(
            task_id=self._task_id,
            current_step=self._state.current_step + 1,
            is_solved=sim.is_solved,
            cumulative_reward=new_cumulative,
            action_history=list(self._state.action_history) + [action],
            last_observation=new_obs,
        )

        done = self._state.is_terminal

        info: dict[str, Any] = {
            "task_id": self._task_id,
            "step": self._state.current_step,
            "is_solved": sim.is_solved,
            "reward_components": sim.reward_components,
            "crash": (
                {"type": sim.crash.crash_type, "recoverable": sim.crash.is_recoverable}
                if sim.crash else None
            ),
        }

        if done:
            grader_result = grade(self._task_id, self._state)
            info["final_grade"] = {
                "score": grader_result.score,
                "passed": grader_result.passed,
                "reason": grader_result.reason,
                "partial_credit": grader_result.partial_credit,
            }

        return StepResult(
            observation=new_obs,
            reward=reward_value,
            reward_reason=reward_reason,
            cumulative_reward=new_cumulative,
            done=done,
            info=info,
        )

    @property
    def state(self) -> MLDebuggerState:
        if self._state is None:
            initial_obs = get_initial_observation(self._task_id)
            return MLDebuggerState(
                task_id=self._task_id,
                current_step=0,
                is_solved=False,
                cumulative_reward=0.0,
                action_history=[],
                last_observation=initial_obs,
            )
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_task_info(self) -> dict[str, Any]:
        return TASK_REGISTRY.get(self._task_id, {})


def _parse_action(raw: dict) -> Any:
    """Deserialize a raw dict into the correct Action subclass."""
    action_type = raw.get("action_type", "")
    if action_type == "tune_hyperparameters":
        return TuneHyperparameters(**raw)
    if action_type == "fix_reshape":
        return FixReshape(**raw)
    if action_type == "add_augmentation":
        return AddAugmentation(**raw)
    if action_type == "adjust_loss_weights":
        return AdjustLossWeights(**raw)
    if action_type == "change_optimizer":               
        return ChangeOptimizer(**raw)         
    if action_type == "toggle_layer_freeze":            
        return ToggleLayerFreeze(**raw)       

    raise ValueError(f"Unknown action_type: '{action_type}'")