"""
tests/test_graders.py

Tests for the deterministic task graders:
  - Easy grader scores correctly for correct/wrong/missing fix
  - Medium grader gives partial credit and penalises NaN crashes
  - Hard grader scores IoU tiers and respects 5-step limit
  - All graders return scores strictly in [0.0, 1.0]
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
from server.tasks import get_initial_observation, grade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(task_id, history=None, step=None, last_obs=None):
    h = history or []
    return MLDebuggerState(
        task_id=task_id,
        current_step=step if step is not None else len(h),
        action_history=h,
        last_observation=last_obs or get_initial_observation(task_id),
    )


def _obs_with_metrics(task_id, train_loss, val_loss, iou=None):
    base = get_initial_observation(task_id)
    metrics = [EpochMetrics(epoch=0, train_loss=train_loss, val_loss=val_loss, iou=iou)]
    return MLDebuggerObservation(
        task_id=base.task_id,
        architecture_summary=base.architecture_summary,
        tensor_shapes=base.tensor_shapes,
        error_trace=base.error_trace,
        metrics_history=metrics,
        step_number=1,
        max_steps=15,
    )


# ---------------------------------------------------------------------------
# Easy grader
# ---------------------------------------------------------------------------

class TestGradeEasy:
    def test_no_action_scores_zero(self):
        result = grade("easy", _state("easy"))
        assert result.score == 0.0
        assert result.passed is False

    def test_wrong_layer_partial_credit(self):
        action = FixReshape(layer="fc1", new_shape=[2304])
        result = grade("easy", _state("easy", history=[action]))
        assert 0.0 < result.score < 1.0
        assert result.passed is False

    def test_correct_layer_wrong_shape_partial_credit(self):
        action = FixReshape(layer="flatten", new_shape=[512])
        result = grade("easy", _state("easy", history=[action]))
        assert result.score > 0.0
        assert result.passed is False
        assert result.partial_credit["correct_layer"] > 0.0
        assert result.partial_credit["correct_shape"] == 0.0

    def test_correct_fix_passes(self):
        action = FixReshape(layer="flatten", new_shape=[2304])
        result = grade("easy", _state("easy", history=[action]))
        assert result.score >= 1.0
        assert result.passed is True

    def test_correct_fix_on_first_step_full_score(self):
        action = FixReshape(layer="flatten", new_shape=[2304])
        result = grade("easy", _state("easy", history=[action], step=1))
        assert result.score == pytest.approx(1.0, abs=0.01)

    def test_wasted_steps_reduce_efficiency(self):
        wrong = FixReshape(layer="fc1", new_shape=[512])
        correct = FixReshape(layer="flatten", new_shape=[2304])
        # 3 wasted steps before correct fix
        history = [wrong, wrong, wrong, correct]
        result = grade("easy", _state("easy", history=history))
        assert result.passed is True
        assert result.partial_credit["efficiency"] < 0.2

    def test_score_in_valid_range(self):
        for history in [
            [],
            [FixReshape(layer="fc1", new_shape=[100])],
            [FixReshape(layer="flatten", new_shape=[2304])],
        ]:
            result = grade("easy", _state("easy", history=history))
            assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Medium grader
# ---------------------------------------------------------------------------

class TestGradeMedium:
    def test_no_action_scores_zero(self):
        result = grade("medium", _state("medium"))
        assert result.score == 0.0
        assert result.passed is False

    def test_irrelevant_strategy_low_score(self):
        action = AddAugmentation(strategy="horizontal_flip")
        result = grade("medium", _state("medium", history=[action]))
        assert result.score < 0.3
        assert result.passed is False

    def test_one_required_strategy_partial_credit(self):
        action = AddAugmentation(strategy="dropout")
        result = grade("medium", _state("medium", history=[action]))
        assert result.partial_credit["strategy_started"] == pytest.approx(0.3)
        assert result.partial_credit["both_strategies"] == 0.0

    def test_both_required_strategies_more_credit(self):
        a1 = AddAugmentation(strategy="dropout")
        a2 = AddAugmentation(strategy="truncate_sequence")
        result = grade("medium", _state("medium", history=[a1, a2]))
        assert result.partial_credit["both_strategies"] == pytest.approx(0.3)

    def test_gap_closed_gives_full_credit(self):
        a1 = AddAugmentation(strategy="dropout")
        a2 = AddAugmentation(strategy="truncate_sequence")
        obs = _obs_with_metrics("medium", train_loss=1.2, val_loss=1.3)  # gap=0.1 < 0.15
        state = _state("medium", history=[a1, a2], last_obs=obs)
        result = grade("medium", state)
        assert result.partial_credit["gap_closed"] == pytest.approx(0.4)
        assert result.passed is True

    def test_nan_penalty_applied(self):
        a1 = AddAugmentation(strategy="dropout")
        a2 = AddAugmentation(strategy="truncate_sequence")
        a3 = TuneHyperparameters(lr=0.9, batch_size=32, epochs=5)  # caused NaN
        obs = _obs_with_metrics("medium", train_loss=1.2, val_loss=1.3)
        state = _state("medium", history=[a1, a2, a3], last_obs=obs)
        result = grade("medium", state)
        # Should have gap_closed but lose 0.1 for NaN
        assert result.score < 1.0

    def test_score_in_valid_range(self):
        for history in [
            [],
            [AddAugmentation(strategy="dropout")],
            [AddAugmentation(strategy="dropout"), AddAugmentation(strategy="truncate_sequence")],
        ]:
            result = grade("medium", _state("medium", history=history))
            assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Hard grader
# ---------------------------------------------------------------------------

class TestGradeHard:
    def test_no_action_baseline_score(self):
        result = grade("hard", _state("hard"))
        # Starting IoU is 0.52 — just above the 0.55 partial credit floor
        assert result.score < 0.5
        assert result.passed is False

    def test_iou_above_085_within_5_steps_passes(self):
        action = AdjustLossWeights(dice_weight=0.65, ce_weight=0.35)
        obs = _obs_with_metrics("hard", train_loss=0.3, val_loss=0.35, iou=0.87)
        state = _state("hard", history=[action], step=3, last_obs=obs)
        result = grade("hard", state)
        assert result.score >= 0.85
        assert result.passed is True

    def test_iou_above_085_after_5_steps_reduced_score(self):
        action = AdjustLossWeights(dice_weight=0.65, ce_weight=0.35)
        obs = _obs_with_metrics("hard", train_loss=0.3, val_loss=0.35, iou=0.87)
        state = _state("hard", history=[action], step=8, last_obs=obs)
        result = grade("hard", state)
        # Should score 0.7 for iou_achievement (exceeded step limit)
        assert result.partial_credit["iou_achievement"] == pytest.approx(0.7)

    def test_iou_075_tier(self):
        obs = _obs_with_metrics("hard", train_loss=0.4, val_loss=0.45, iou=0.76)
        state = _state("hard", last_obs=obs)
        result = grade("hard", state)
        assert result.partial_credit["iou_achievement"] == pytest.approx(0.5)
        assert result.passed is False

    def test_iou_065_tier(self):
        obs = _obs_with_metrics("hard", train_loss=0.5, val_loss=0.55, iou=0.67)
        state = _state("hard", last_obs=obs)
        result = grade("hard", state)
        assert result.partial_credit["iou_achievement"] == pytest.approx(0.3)

    def test_optimal_lr_gives_config_bonus(self):
        action = TuneHyperparameters(lr=0.001, batch_size=16, epochs=5)  # in optimal range
        state = _state("hard", history=[action])
        result = grade("hard", state)
        assert result.partial_credit["config_quality"] >= 0.1

    def test_optimal_dice_gives_config_bonus(self):
        action = AdjustLossWeights(dice_weight=0.65, ce_weight=0.35)  # in optimal range
        state = _state("hard", history=[action])
        result = grade("hard", state)
        assert result.partial_credit["config_quality"] >= 0.1

    def test_score_in_valid_range(self):
        for iou in [0.52, 0.60, 0.70, 0.80, 0.90]:
            obs = _obs_with_metrics("hard", train_loss=0.4, val_loss=0.45, iou=iou)
            state = _state("hard", last_obs=obs)
            result = grade("hard", state)
            assert 0.0 <= result.score <= 1.0