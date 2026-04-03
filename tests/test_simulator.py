"""
tests/test_simulator.py

Tests for the fast-simulation engine:
  - detect_crash() catches correct error conditions per task
  - compute_loss_curve() produces valid, deterministic metrics
  - simulate_step() returns correct SimulationResult shape
  - Crash penalties and solve conditions are correctly set
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import (
    AddAugmentation,
    AdjustLossWeights,
    EpochMetrics,
    FixReshape,
    MLDebuggerObservation,
    MLDebuggerState,
    TuneHyperparameters,
)
from server.simulator import (
    CrashResult,
    SimulationResult,
    compute_loss_curve,
    detect_crash,
    simulate_step,
)
from server.tasks import get_initial_observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(task_id, history=None, last_obs=None):
    return MLDebuggerState(
        task_id=task_id,
        current_step=len(history) if history else 0,
        action_history=history or [],
        last_observation=last_obs or get_initial_observation(task_id),
    )


# ---------------------------------------------------------------------------
# detect_crash — Easy task
# ---------------------------------------------------------------------------

class TestDetectCrashEasy:
    def test_no_crash_on_non_reshape_action(self):
        state = _make_state("easy")
        action = TuneHyperparameters(lr=0.01, batch_size=32, epochs=5)
        assert detect_crash("easy", action, state) is None

    def test_crash_wrong_layer(self):
        state = _make_state("easy")
        action = FixReshape(layer="fc1", new_shape=[2304])
        result = detect_crash("easy", action, state)
        assert result is not None
        assert result.crash_type == "shape_mismatch"
        assert "fc1" in result.error_trace

    def test_crash_wrong_shape(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[512])
        result = detect_crash("easy", action, state)
        assert result is not None
        assert result.crash_type == "shape_mismatch"
        assert "2304" in result.error_trace

    def test_no_crash_correct_fix(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[2304])
        assert detect_crash("easy", action, state) is None


# ---------------------------------------------------------------------------
# detect_crash — Medium task
# ---------------------------------------------------------------------------

class TestDetectCrashMedium:
    def test_no_crash_valid_lr(self):
        state = _make_state("medium")
        action = TuneHyperparameters(lr=0.001, batch_size=16, epochs=5)
        assert detect_crash("medium", action, state) is None

    def test_nan_crash_extreme_lr(self):
        state = _make_state("medium")
        action = TuneHyperparameters(lr=0.9, batch_size=16, epochs=5)
        result = detect_crash("medium", action, state)
        assert result is not None
        assert result.crash_type == "nan_loss"
        assert "NaN" in result.error_trace

    def test_no_crash_augmentation(self):
        state = _make_state("medium")
        action = AddAugmentation(strategy="dropout")
        assert detect_crash("medium", action, state) is None


# ---------------------------------------------------------------------------
# detect_crash — Hard task
# ---------------------------------------------------------------------------

class TestDetectCrashHard:
    def test_no_crash_balanced_weights(self):
        state = _make_state("hard")
        action = AdjustLossWeights(dice_weight=0.6, ce_weight=0.4)
        assert detect_crash("hard", action, state) is None

    def test_crash_extreme_dice_weight(self):
        state = _make_state("hard")
        action = AdjustLossWeights(dice_weight=0.98, ce_weight=0.02)
        result = detect_crash("hard", action, state)
        assert result is not None
        assert result.crash_type == "oom"

    def test_crash_extreme_ce_weight(self):
        state = _make_state("hard")
        action = AdjustLossWeights(dice_weight=0.02, ce_weight=0.98)
        result = detect_crash("hard", action, state)
        assert result is not None
        assert result.crash_type == "oom"


# ---------------------------------------------------------------------------
# detect_crash — Infinite loop detection (all tasks)
# ---------------------------------------------------------------------------

class TestInfiniteLoopDetection:
    def test_loop_detected_after_3_identical_actions(self):
        action = FixReshape(layer="flatten", new_shape=[512])
        state = _make_state("easy", history=[action, action, action])
        result = detect_crash("easy", action, state)
        assert result is not None
        assert result.crash_type == "invalid_action"
        assert "InfiniteLoop" in result.error_trace

    def test_no_loop_with_different_actions(self):
        a1 = FixReshape(layer="flatten", new_shape=[512])
        a2 = FixReshape(layer="flatten", new_shape=[1024])
        a3 = FixReshape(layer="flatten", new_shape=[2048])
        state = _make_state("easy", history=[a1, a2, a3])
        action = FixReshape(layer="flatten", new_shape=[2304])
        assert detect_crash("easy", action, state) is None

    def test_no_loop_with_only_2_identical(self):
        action = FixReshape(layer="fc1", new_shape=[512])
        state = _make_state("easy", history=[action, action])
        assert detect_crash("easy", action, state) is None


# ---------------------------------------------------------------------------
# compute_loss_curve
# ---------------------------------------------------------------------------

class TestComputeLossCurve:
    def test_easy_correct_fix_returns_metrics(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[2304])
        metrics = compute_loss_curve("easy", action, state, starting_epoch=0)
        assert len(metrics) == 3
        for m in metrics:
            assert m.train_loss >= 0
            assert m.val_loss >= 0
            assert m.accuracy is not None

    def test_easy_wrong_fix_returns_empty(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[512])
        metrics = compute_loss_curve("easy", action, state, starting_epoch=0)
        assert metrics == []

    def test_medium_returns_metrics(self):
        state = _make_state("medium")
        action = AddAugmentation(strategy="dropout")
        metrics = compute_loss_curve("medium", action, state, starting_epoch=2)
        assert len(metrics) > 0
        assert all(m.epoch >= 2 for m in metrics)

    def test_hard_returns_iou_metrics(self):
        state = _make_state("hard")
        action = AdjustLossWeights(dice_weight=0.65, ce_weight=0.35)
        metrics = compute_loss_curve("hard", action, state, starting_epoch=1)
        assert len(metrics) == 3
        assert all(m.iou is not None for m in metrics)
        assert all(0.0 <= m.iou <= 1.0 for m in metrics)

    def test_determinism(self):
        """Same inputs must always produce identical outputs."""
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[2304])
        m1 = compute_loss_curve("easy", action, state, starting_epoch=0)
        m2 = compute_loss_curve("easy", action, state, starting_epoch=0)
        assert [m.train_loss for m in m1] == [m.train_loss for m in m2]
        assert [m.val_loss for m in m1] == [m.val_loss for m in m2]

    def test_epoch_indices_are_sequential(self):
        state = _make_state("medium")
        action = TuneHyperparameters(lr=0.001, batch_size=32, epochs=5)
        metrics = compute_loss_curve("medium", action, state, starting_epoch=3)
        epochs = [m.epoch for m in metrics]
        assert epochs == list(range(3, 3 + len(metrics)))


# ---------------------------------------------------------------------------
# simulate_step — full integration
# ---------------------------------------------------------------------------

class TestSimulateStep:
    def test_crash_result_has_no_metrics(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[512])
        result = simulate_step("easy", action, state)
        assert result.crash is not None
        assert result.new_metrics == []
        assert result.is_solved is False

    def test_crash_penalty_in_reward_components(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[512])
        result = simulate_step("easy", action, state)
        assert result.reward_components["crash_penalty"] == -0.5

    def test_loop_penalty_is_minus_03(self):
        action = FixReshape(layer="fc1", new_shape=[512])
        state = _make_state("easy", history=[action, action, action])
        result = simulate_step("easy", action, state)
        assert result.reward_components["crash_penalty"] == -0.3

    def test_correct_easy_fix_is_solved(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[2304])
        result = simulate_step("easy", action, state)
        assert result.crash is None
        assert result.is_solved is True
        assert result.reward_components["solve_bonus"] == 2.0
        assert len(result.new_metrics) == 3

    def test_epoch_bonus_in_reward_components(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[2304])
        result = simulate_step("easy", action, state)
        assert result.reward_components["epoch_bonus"] == pytest.approx(0.3, abs=0.01)

    def test_architecture_summary_changes_after_fix(self):
        state = _make_state("easy")
        action = FixReshape(layer="flatten", new_shape=[2304])
        result = simulate_step("easy", action, state)
        assert "FIXED" in result.architecture_summary

    def test_architecture_summary_shows_broken_before_fix(self):
        state = _make_state("easy")
        action = TuneHyperparameters(lr=0.001, batch_size=32, epochs=5)
        result = simulate_step("easy", action, state)
        assert "BROKEN" in result.architecture_summary

    def test_hard_iou_increases_with_optimal_config(self):
        state = _make_state("hard")
        action = AdjustLossWeights(dice_weight=0.65, ce_weight=0.35)
        result = simulate_step("hard", action, state)
        assert result.new_metrics[-1].iou > 0.52

    def test_medium_val_gap_closes_with_strategies(self):
        state = _make_state("medium")
        a1 = AddAugmentation(strategy="dropout")
        state2 = MLDebuggerState(
            task_id="medium",
            current_step=1,
            action_history=[a1],
            last_observation=get_initial_observation("medium"),
        )
        a2 = AddAugmentation(strategy="truncate_sequence")
        result = simulate_step("medium", a2, state2)
        assert result.new_metrics[-1].val_loss < 2.40  # gap should close