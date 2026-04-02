"""
server/reward.py — Dense Reward Function

Reward schedule:
  +0.1  per successful simulated epoch
  +1.0  per 0.05 drop in validation loss
  -0.5  crash / OOM / NaN
  -0.3  infinite loop (repeated failed action)
  +2.0  task solved bonus
"""

from __future__ import annotations

from server.simulator import SimulationResult


def compute_reward(result: SimulationResult, cumulative_so_far: float) -> tuple[float, str]:
    """
    Returns (reward_value, reason_string).
    Clamped to [-1.0, 2.0].
    """
    c = result.reward_components
    reasons: list[str] = []
    value = 0.0

    if c["crash_penalty"] < 0:
        value += c["crash_penalty"]
        reasons.append(
            f"Crash penalty: {c['crash_penalty']:.1f}"
            if c["crash_penalty"] == -0.5
            else f"Loop penalty: {c['crash_penalty']:.1f}"
        )

    if c["epoch_bonus"] > 0:
        value += c["epoch_bonus"]
        n = round(c["epoch_bonus"] / 0.1)
        reasons.append(f"{n} epoch(s) simulated: +{c['epoch_bonus']:.1f}")

    if c["loss_improvement"] > 0:
        value += c["loss_improvement"]
        reasons.append(f"Val loss improved: +{c['loss_improvement']:.2f}")

    if c["solve_bonus"] > 0:
        value += c["solve_bonus"]
        reasons.append(f"Task SOLVED: +{c['solve_bonus']:.1f}")

    value = max(-1.0, min(2.0, value))
    reason = "; ".join(reasons) if reasons else "No progress this step."
    return round(value, 6), reason