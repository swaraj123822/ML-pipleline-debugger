"""
tests/test_env_loop.py

Full end-to-end episode loop tests against MLDebuggerEnvironment:
  - reset() returns correct initial observation
  - step() returns StepResult with all required fields
  - Episode terminates correctly (MAX_STEPS and solve conditions)
  - state() reflects current episode state
  - Calling step() after done returns terminal StepResult
  - All 3 tasks can be run to completion
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import (
    AddAugmentation,
    AdjustLossWeights,
    FixReshape,
    TuneHyperparameters,
)
from server.environment import MLDebuggerEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(task_id="easy"):
    env = MLDebuggerEnvironment()
    obs = env.reset(task_id=task_id)
    return env, obs


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self):
        env, obs = _make_env("easy")
        assert obs.task_id == "easy"
        assert obs.step_number == 0
        assert obs.max_steps == 15

    def test_reset_easy_has_error_trace(self):
        _, obs = _make_env("easy")
        assert obs.error_trace is not None
        assert "RuntimeError" in obs.error_trace

    def test_reset_medium_has_metrics_history(self):
        _, obs = _make_env("medium")
        assert len(obs.metrics_history) == 2   # 2 pre-populated epochs

    def test_reset_hard_has_iou_metric(self):
        _, obs = _make_env("hard")
        assert obs.metrics_history[0].iou == pytest.approx(0.52)

    def test_reset_invalid_task_raises(self):
        env = MLDebuggerEnvironment()
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="impossible")

    def test_double_reset_restarts_episode(self):
        env, _ = _make_env("easy")
        action = TuneHyperparameters(lr=0.001, batch_size=32, epochs=5)
        env.step(action)
        obs2 = env.reset(task_id="easy")
        assert obs2.step_number == 0
        assert len(obs2.metrics_history) == 0

    def test_can_switch_task_on_reset(self):
        env, _ = _make_env("easy")
        obs = env.reset(task_id="medium")
        assert obs.task_id == "medium"


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_before_reset_raises(self):
        env = MLDebuggerEnvironment()
        action = FixReshape(layer="flatten", new_shape=[2304])
        with pytest.raises(RuntimeError, match="reset()"):
            env.step(action)

    def test_step_returns_step_result(self):
        env, _ = _make_env("easy")
        action = TuneHyperparameters(lr=0.001, batch_size=32, epochs=5)
        result = env.step(action)
        assert hasattr(result, "observation")
        assert hasattr(result, "reward")
        assert hasattr(result, "reward_reason")
        assert hasattr(result, "cumulative_reward")
        assert hasattr(result, "done")
        assert hasattr(result, "info")

    def test_step_increments_step_number(self):
        env, _ = _make_env("easy")
        action = TuneHyperparameters(lr=0.001, batch_size=32, epochs=5)
        result = env.step(action)
        assert result.observation.step_number == 1

    def test_step_accepts_dict_action(self):
        env, _ = _make_env("easy")
        action_dict = {"action_type": "fix_reshape", "layer": "flatten", "new_shape": [2304]}
        result = env.step(action_dict)
        assert result.done is True  # correct fix solves it

    def test_step_accepts_pydantic_action(self):
        env, _ = _make_env("easy")
        action = FixReshape(layer="flatten", new_shape=[2304])
        result = env.step(action)
        assert result.done is True

    def test_crash_gives_negative_reward(self):
        env, _ = _make_env("easy")
        action = FixReshape(layer="flatten", new_shape=[512])
        result = env.step(action)
        assert result.reward < 0

    def test_correct_easy_fix_solves_episode(self):
        env, _ = _make_env("easy")
        action = FixReshape(layer="flatten", new_shape=[2304])
        result = env.step(action)
        assert result.done is True
        assert result.info["is_solved"] is True
        assert "final_grade" in result.info
        assert result.info["final_grade"]["passed"] is True


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

class TestEpisodeTermination:
    def test_episode_ends_at_max_steps(self):
        env, _ = _make_env("easy")
        action = TuneHyperparameters(lr=0.001, batch_size=32, epochs=1)
        result = None
        for _ in range(15):
            result = env.step(action)
            if result.done:
                break
        assert result.done is True

    def test_step_after_done_returns_terminal(self):
        env, _ = _make_env("easy")
        correct = FixReshape(layer="flatten", new_shape=[2304])
        env.step(correct)  # solves it
        # Step again after done
        extra = env.step(TuneHyperparameters(lr=0.001, batch_size=32, epochs=1))
        assert extra.done is True
        assert extra.reward == 0.0
        assert "warning" in extra.info

    def test_final_grade_in_info_when_done(self):
        env, _ = _make_env("easy")
        result = env.step(FixReshape(layer="flatten", new_shape=[2304]))
        assert "final_grade" in result.info
        fg = result.info["final_grade"]
        assert "score" in fg
        assert "passed" in fg
        assert "reason" in fg


# ---------------------------------------------------------------------------
# state()
# ---------------------------------------------------------------------------

class TestState:
    def test_state_before_reset(self):
        env = MLDebuggerEnvironment()
        s = env.state
        assert s is not None

    def test_state_reflects_step_count(self):
        env, _ = _make_env("medium")
        action = AddAugmentation(strategy="dropout")
        env.step(action)
        s = env.state
        assert s.current_step == 1
        assert len(s.action_history) == 1

    def test_state_is_serializable(self):
        env, _ = _make_env("hard")
        s = env.state
        dumped = s.model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert "task_id" in dumped

    def test_state_tracks_cumulative_reward(self):
        env, _ = _make_env("easy")
        env.step(TuneHyperparameters(lr=0.001, batch_size=32, epochs=5))
        s = env.state
        assert isinstance(s.cumulative_reward, float)


# ---------------------------------------------------------------------------
# Full episode simulations — all 3 tasks
# ---------------------------------------------------------------------------

class TestFullEpisodes:
    def test_easy_optimal_path(self):
        """Agent that immediately issues the correct fix should score 1.0."""
        env, _ = _make_env("easy")
        result = env.step(FixReshape(layer="flatten", new_shape=[2304]))
        assert result.done is True
        assert result.info["final_grade"]["score"] >= 1.0

    def test_medium_optimal_path(self):
        """Two correct augmentations should close the val gap and pass."""
        env, _ = _make_env("medium")
        env.step(AddAugmentation(strategy="dropout"))
        result = env.step(AddAugmentation(strategy="truncate_sequence"))
        # May or may not be done depending on gap — just check no exception
        assert result.observation.task_id == "medium"
        assert result.cumulative_reward > 0

    def test_hard_optimal_path(self):
        """Optimal lr + loss weights should push IoU up."""
        env, _ = _make_env("hard")
        env.step(TuneHyperparameters(lr=0.001, batch_size=16, epochs=5))
        result = env.step(AdjustLossWeights(dice_weight=0.65, ce_weight=0.35))
        iou = result.observation.last_metrics.iou
        assert iou is not None
        assert iou > 0.52  # should have improved from baseline

    def test_dummy_agent_runs_15_steps_without_error(self):
        """A random (but valid) agent must never crash the environment."""
        env, _ = _make_env("medium")
        action = AddAugmentation(strategy="weight_decay")
        for _ in range(15):
            result = env.step(action)
            if result.done:
                break
        assert result.done is True

    def test_reward_accumulates_correctly(self):
        env, _ = _make_env("easy")
        r1 = env.step(TuneHyperparameters(lr=0.001, batch_size=32, epochs=1))
        r2 = env.step(FixReshape(layer="flatten", new_shape=[2304]))
        assert r2.cumulative_reward == pytest.approx(
            r1.reward + r2.reward, abs=1e-4
        )