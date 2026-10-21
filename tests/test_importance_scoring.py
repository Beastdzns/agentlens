"""
Unit tests for ``agentlens.observation.importance``.

Covers:
- ``ScoringConfig`` defaults and coverage of every status / event type.
- Base status + event-type scoring (errors and retries high, benign repeats low).
- Additive modifiers: parent, latency, exception.
- Repetition penalty, its cap, and its exemptions.
- Clamping to ``[0, 1]``.
- Determinism and custom configuration.
- The relationship between the score and the policy thresholds
  (``0.60`` FULL / ``0.35`` COMPRESSED / ``0.15`` METADATA_ONLY).

The scorer is pure, so these are fast unit tests with no I/O.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from uuid import uuid4

import pytest

from agentlens.events import EventStatus, EventType
from agentlens.observation import (
    DEFAULT_SCORING_CONFIG,
    ImportanceScorer,
    ScoringConfig,
)

# A non-null parent id; used to switch the parent bonus off in isolation tests.
PARENT = uuid4()

# Policy thresholds (see agentlens.observation.policy.PolicyConfig).
COMPRESS_THRESHOLD = 0.60
METADATA_THRESHOLD = 0.35
DROP_THRESHOLD = 0.15


def _flat_config(**overrides: Any) -> ScoringConfig:
    """
    Config with equal base/weight so a single rule can be tested in isolation.

    Every status contributes ``0.5`` and every event type ``0.5`` by default,
    which averages to a neutral score of ``0.5`` before any modifier.
    """
    params: dict[str, Any] = {
        "base_by_status": dict.fromkeys(EventStatus, 0.5),
        "weight_by_type": dict.fromkeys(EventType, 0.5),
        "parent_bonus": 0.0,
        "latency_bonus": 0.0,
        "exception_bonus": 0.0,
        "repetition_penalty": 0.2,
        "repetition_cap": 2,
    }
    params.update(overrides)
    return ScoringConfig(**params)


# ============================================================================
# ScoringConfig
# ============================================================================


class TestScoringConfig:
    """Test configuration defaults and coverage."""

    def test_default_config_is_scoring_config(self) -> None:
        assert isinstance(DEFAULT_SCORING_CONFIG, ScoringConfig)

    def test_default_factory_matches_shared_default(self) -> None:
        assert ScoringConfig() == DEFAULT_SCORING_CONFIG

    def test_default_status_map_covers_all_statuses(self) -> None:
        assert set(DEFAULT_SCORING_CONFIG.base_by_status) == set(EventStatus)

    def test_default_type_map_covers_all_event_types(self) -> None:
        assert set(DEFAULT_SCORING_CONFIG.weight_by_type) == set(EventType)

    def test_default_tunable_values(self) -> None:
        cfg = DEFAULT_SCORING_CONFIG
        assert cfg.parent_bonus == pytest.approx(0.10)
        assert cfg.latency_high_ms == pytest.approx(1000.0)
        assert cfg.latency_bonus == pytest.approx(0.10)
        assert cfg.exception_bonus == pytest.approx(0.10)
        assert cfg.repetition_penalty == pytest.approx(0.15)
        assert cfg.repetition_cap == 3
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.max_score == pytest.approx(1.0)

    def test_config_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            DEFAULT_SCORING_CONFIG.parent_bonus = 0.5  # type: ignore[misc]


# ============================================================================
# Basics
# ============================================================================


class TestScorerBasics:
    """Test the scorer surface and determinism."""

    def test_config_property_exposed(self) -> None:
        scorer = ImportanceScorer(_flat_config())
        assert scorer.config.weight_by_type[EventType.LLM_CALL] == pytest.approx(0.5)

    def test_none_config_uses_default(self) -> None:
        assert ImportanceScorer().config == DEFAULT_SCORING_CONFIG

    def test_score_is_float_in_range(self, sample_event) -> None:
        score = ImportanceScorer().score(sample_event())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic_same_input(self, sample_event) -> None:
        scorer = ImportanceScorer()
        event = sample_event(event_type=EventType.TOOL_CALL)
        assert scorer.score(event) == scorer.score(event)

    def test_deterministic_independent_of_call_order(self, sample_event) -> None:
        scorer = ImportanceScorer()
        a = sample_event(event_type=EventType.RETRIEVAL)
        b = sample_event(event_type=EventType.MEMORY_WRITE)
        first = scorer.score(a)
        scorer.score(b)
        assert scorer.score(a) == first


# ============================================================================
# Base scoring (default config)
# ============================================================================


class TestBaseScoring:
    """Test status- and type-driven scoring with the default config."""

    def test_success_llm_call_baseline(self, sample_event) -> None:
        # (0.30 success + 0.50 llm_call) / 2 + 0.10 root (no parent)
        event = sample_event(event_type=EventType.LLM_CALL, status=EventStatus.SUCCESS)
        assert ImportanceScorer().score(event) == pytest.approx(0.50)

    def test_failure_scores_above_success(self, sample_event) -> None:
        scorer = ImportanceScorer()
        failed = sample_event(
            event_type=EventType.LLM_CALL, status=EventStatus.FAILURE, parent_id=PARENT
        )
        ok = sample_event(
            event_type=EventType.LLM_CALL, status=EventStatus.SUCCESS, parent_id=PARENT
        )
        assert scorer.score(failed) > scorer.score(ok)

    def test_error_event_scores_highest(self, sample_event) -> None:
        scorer = ImportanceScorer()
        err = sample_event(event_type=EventType.ERROR, parent_id=PARENT)
        llm = sample_event(event_type=EventType.LLM_CALL, parent_id=PARENT)
        assert scorer.score(err) == pytest.approx(0.65)
        assert scorer.score(err) > scorer.score(llm)

    def test_retry_event_scores_high(self, sample_event) -> None:
        scorer = ImportanceScorer()
        retry = sample_event(event_type=EventType.RETRY, parent_id=PARENT)
        llm = sample_event(event_type=EventType.LLM_CALL, parent_id=PARENT)
        assert scorer.score(retry) == pytest.approx(0.575)
        assert scorer.score(retry) > scorer.score(llm)

    def test_memory_read_scores_low(self, sample_event) -> None:
        scorer = ImportanceScorer()
        mem = sample_event(event_type=EventType.MEMORY_READ, parent_id=PARENT)
        llm = sample_event(event_type=EventType.LLM_CALL, parent_id=PARENT)
        assert scorer.score(mem) == pytest.approx(0.325)
        assert scorer.score(mem) < scorer.score(llm)

    def test_base_and_weight_are_averaged(self, sample_event) -> None:
        """The score is the mean of the status base and the type weight."""
        scorer = ImportanceScorer()
        event = sample_event(
            event_type=EventType.TOOL_CALL,
            status=EventStatus.PARTIAL,
            parent_id=PARENT,
            latency_ms=10.0,
        )
        # (0.70 partial + 0.55 tool_call) / 2
        assert scorer.score(event) == pytest.approx(0.625)


class TestScoreThresholdAlignment:
    """The score distribution must line up with the frozen policy thresholds."""

    def test_error_reaches_full_threshold(self, sample_event) -> None:
        event = sample_event(event_type=EventType.ERROR, parent_id=PARENT)
        assert ImportanceScorer().score(event) >= COMPRESS_THRESHOLD

    def test_failure_reaches_full_threshold(self, sample_event) -> None:
        event = sample_event(
            event_type=EventType.LLM_CALL, status=EventStatus.FAILURE, parent_id=PARENT
        )
        assert ImportanceScorer().score(event) >= COMPRESS_THRESHOLD

    def test_final_response_reaches_full_threshold(self, sample_event) -> None:
        event = sample_event(event_type=EventType.FINAL_RESPONSE, parent_id=PARENT)
        assert ImportanceScorer().score(event) >= COMPRESS_THRESHOLD

    def test_benign_success_stays_below_full_threshold(self, sample_event) -> None:
        """Ordinary successful steps must be reducible, else adaptive is a no-op."""
        event = sample_event(event_type=EventType.LLM_CALL, parent_id=PARENT)
        score = ImportanceScorer().score(event)
        assert score < COMPRESS_THRESHOLD
        assert score >= METADATA_THRESHOLD

    def test_repetitive_memory_read_falls_below_metadata_threshold(self, sample_event) -> None:
        """A repeated low-value step drops out of the COMPRESSED band."""
        event = sample_event(event_type=EventType.MEMORY_READ, parent_id=PARENT)
        score = ImportanceScorer().score(event, repeat_index=1)
        assert score < METADATA_THRESHOLD
        assert score >= DROP_THRESHOLD

    def test_heavily_repeated_step_reaches_drop_band(self, sample_event) -> None:
        """Repeated enough, a low-value step falls into the DROP band."""
        event = sample_event(event_type=EventType.MEMORY_READ, parent_id=PARENT)
        score = ImportanceScorer().score(event, repeat_index=3)
        assert score < DROP_THRESHOLD


# ============================================================================
# Modifiers
# ============================================================================


class TestModifiers:
    """Test parent / latency / exception bonuses."""

    def test_parent_bonus_applied_when_no_parent(self, sample_event) -> None:
        scorer = ImportanceScorer()
        root = sample_event(parent_id=None)
        child = sample_event(parent_id=PARENT)
        assert scorer.score(root) - scorer.score(child) == pytest.approx(0.10)

    def test_parent_bonus_absent_when_parent_set(self, sample_event) -> None:
        event = sample_event(parent_id=PARENT)
        assert ImportanceScorer().score(event) == pytest.approx(0.40)

    def test_latency_bonus_at_threshold(self, sample_event) -> None:
        event = sample_event(parent_id=PARENT, latency_ms=1000.0)
        assert ImportanceScorer().score(event) == pytest.approx(0.50)

    def test_latency_bonus_above_threshold(self, sample_event) -> None:
        event = sample_event(parent_id=PARENT, latency_ms=5000.0)
        assert ImportanceScorer().score(event) == pytest.approx(0.50)

    def test_no_latency_bonus_below_threshold(self, sample_event) -> None:
        event = sample_event(parent_id=PARENT, latency_ms=999.0)
        assert ImportanceScorer().score(event) == pytest.approx(0.40)

    def test_no_latency_bonus_when_latency_missing(self, sample_event) -> None:
        event = sample_event(parent_id=PARENT, latency_ms=None)
        assert ImportanceScorer().score(event) == pytest.approx(0.40)

    def test_exception_bonus_applied(self, sample_event) -> None:
        event = sample_event(
            parent_id=PARENT,
            latency_ms=100.0,
            metadata={"exception_type": "ValueError"},
        )
        assert ImportanceScorer().score(event) == pytest.approx(0.50)

    def test_no_exception_bonus_without_metadata(self, sample_event) -> None:
        event = sample_event(parent_id=PARENT, latency_ms=100.0, metadata=None)
        assert ImportanceScorer().score(event) == pytest.approx(0.40)

    def test_no_exception_bonus_when_key_absent(self, sample_event) -> None:
        event = sample_event(
            parent_id=PARENT, latency_ms=100.0, metadata={"other": "value"}
        )
        assert ImportanceScorer().score(event) == pytest.approx(0.40)

    def test_modifiers_are_additive(self, sample_event) -> None:
        scorer = ImportanceScorer(
            _flat_config(
                parent_bonus=0.10,
                latency_bonus=0.05,
                exception_bonus=0.03,
            )
        )
        event = sample_event(
            parent_id=None,
            latency_ms=5000.0,
            metadata={"exception_type": "TimeoutError"},
        )
        # 0.5 (neutral) + 0.10 + 0.05 + 0.03
        assert scorer.score(event) == pytest.approx(0.68)


# ============================================================================
# Repetition penalty
# ============================================================================


class TestRepetitionPenalty:
    """Test the repetition penalty, its cap, and its exemptions."""

    def test_first_occurrence_not_penalised(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config())
        event = sample_event(parent_id=PARENT, latency_ms=10.0)
        assert scorer.score(event, repeat_index=0) == pytest.approx(0.50)

    def test_repeated_success_penalised(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config())
        event = sample_event(parent_id=PARENT, latency_ms=10.0)
        assert scorer.score(event, repeat_index=1) == pytest.approx(0.30)

    def test_repetition_penalty_is_capped(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config(repetition_penalty=0.2, repetition_cap=2))
        event = sample_event(parent_id=PARENT, latency_ms=10.0)
        # 0.5 - 0.2 * min(5, 2) = 0.1
        assert scorer.score(event, repeat_index=5) == pytest.approx(0.10)

    def test_custom_penalty_and_cap(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config(repetition_penalty=0.1, repetition_cap=1))
        event = sample_event(parent_id=PARENT, latency_ms=10.0)
        assert scorer.score(event, repeat_index=1) == pytest.approx(0.40)
        assert scorer.score(event, repeat_index=9) == pytest.approx(0.40)

    def test_failure_not_penalised(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config())
        event = sample_event(
            status=EventStatus.FAILURE, parent_id=PARENT, latency_ms=10.0
        )
        assert scorer.score(event, repeat_index=5) == pytest.approx(0.50)

    def test_final_response_not_penalised(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config())
        event = sample_event(
            event_type=EventType.FINAL_RESPONSE, parent_id=PARENT, latency_ms=10.0
        )
        assert scorer.score(event, repeat_index=5) == pytest.approx(0.50)

    def test_decision_not_penalised(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config())
        event = sample_event(
            event_type=EventType.DECISION, parent_id=PARENT, latency_ms=10.0
        )
        assert scorer.score(event, repeat_index=5) == pytest.approx(0.50)

    def test_non_success_not_penalised(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config())
        event = sample_event(
            status=EventStatus.PENDING, parent_id=PARENT, latency_ms=10.0
        )
        assert scorer.score(event, repeat_index=5) == pytest.approx(0.50)

    def test_negative_repeat_index_treated_as_first(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config())
        event = sample_event(parent_id=PARENT, latency_ms=10.0)
        assert scorer.score(event, repeat_index=-3) == pytest.approx(0.50)


# ============================================================================
# Clamping
# ============================================================================


class TestClamping:
    """Test score clamping to the configured range."""

    def test_high_scores_clamp_to_one(self, sample_event) -> None:
        event = sample_event(event_type=EventType.ERROR, status=EventStatus.FAILURE)
        assert ImportanceScorer().score(event) == pytest.approx(1.0)

    def test_low_scores_clamp_to_zero(self, sample_event) -> None:
        scorer = ImportanceScorer(
            _flat_config(
                base_by_status=dict.fromkeys(EventStatus, 0.0),
                weight_by_type=dict.fromkeys(EventType, 0.0),
                repetition_penalty=0.5,
                repetition_cap=3,
            )
        )
        event = sample_event(parent_id=PARENT, latency_ms=10.0)
        assert scorer.score(event, repeat_index=1) == pytest.approx(0.0)

    def test_custom_max_score(self, sample_event) -> None:
        scorer = ImportanceScorer(
            _flat_config(
                base_by_status=dict.fromkeys(EventStatus, 1.0),
                weight_by_type=dict.fromkeys(EventType, 0.9),
                max_score=0.75,
            )
        )
        event = sample_event(parent_id=PARENT, latency_ms=10.0)
        assert scorer.score(event) == pytest.approx(0.75)


# ============================================================================
# Custom configuration
# ============================================================================


class TestCustomConfig:
    """Test that custom configuration is respected."""

    def test_custom_status_base_respected(self, sample_event) -> None:
        scorer = ImportanceScorer(
            _flat_config(
                base_by_status=dict.fromkeys(EventStatus, 0.0) | {EventStatus.SUCCESS: 0.4},
                weight_by_type=dict.fromkeys(EventType, 0.0),
            )
        )
        event = sample_event(parent_id=PARENT, latency_ms=10.0)
        assert scorer.score(event) == pytest.approx(0.20)

    def test_custom_type_weight_respected(self, sample_event) -> None:
        scorer = ImportanceScorer(
            _flat_config(
                weight_by_type=dict.fromkeys(EventType, 0.5) | {EventType.LLM_CALL: 0.9}
            )
        )
        event = sample_event(
            event_type=EventType.LLM_CALL, parent_id=PARENT, latency_ms=10.0
        )
        assert scorer.score(event) == pytest.approx(0.70)

    def test_custom_parent_bonus_respected(self, sample_event) -> None:
        scorer = ImportanceScorer(_flat_config(parent_bonus=0.25))
        root = sample_event(parent_id=None, latency_ms=10.0)
        child = sample_event(parent_id=PARENT, latency_ms=10.0)
        assert scorer.score(root) == pytest.approx(0.75)
        assert scorer.score(child) == pytest.approx(0.50)
