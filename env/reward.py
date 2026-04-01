"""
Dense Reward Function

Provides continuous reward signal over the full episode trajectory.
Called by MLDebuggerEnv.step() after simulate_step() returns.

Reward schedule (from execution plan):
  +0.1  per successful (non-crashing) simulated epoch
  +1.0  for every 0.05 drop in validation loss
  -0.5  for actions causing crash, OOM, or NaN loss
  -0.3  for infinite-loop detection (repeated failed action)
  +2.0  task solved bonus (added once on completion)

All values clamped to [-1.0, 2.0] per Reward model validator.
"""

from __future__ import annotations

from env.models import Reward
from env.simulator import SimulationResult


def compute_reward(
    result: SimulationResult,
    cumulative_so_far: float,
) -> Reward:
    """
    Compute the Reward for a single step from a SimulationResult.

    Args:
        result:            Output of simulator.simulate_step()
        cumulative_so_far: Running total before this step

    Returns:
        A fully populated Reward model with value, reason, and cumulative.
    """
    components = result.reward_components
    reasons: list[str] = []

    value = 0.0

    # --- Crash / loop penalties ---
    if components["crash_penalty"] < 0:
        value += components["crash_penalty"]
        if components["crash_penalty"] == -0.5:
            reasons.append("Pipeline crashed (shape mismatch / NaN / OOM): -0.5")
        else:
            reasons.append("Repeated failed action detected (loop): -0.3")

    # --- Epoch bonus ---
    if components["epoch_bonus"] > 0:
        value += components["epoch_bonus"]
        n_epochs = round(components["epoch_bonus"] / 0.1)
        reasons.append(f"{n_epochs} successful epoch(s) simulated: +{components['epoch_bonus']:.1f}")

    # --- Loss improvement bonus ---
    if components["loss_improvement"] > 0:
        value += components["loss_improvement"]
        reasons.append(
            f"Validation loss improved "
            f"(+{components['loss_improvement']:.2f} reward from loss drop)"
        )

    # --- Solve bonus ---
    if components["solve_bonus"] > 0:
        value += components["solve_bonus"]
        reasons.append(f"Task SOLVED! Bonus: +{components['solve_bonus']:.1f}")

    # Build reason string
    reason = "; ".join(reasons) if reasons else "No progress this step."

    # Clamp handled by Reward validator
    new_cumulative = round(cumulative_so_far + value, 6)

    return Reward(
        value=value,
        reason=reason,
        cumulative=new_cumulative,
    )





