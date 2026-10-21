"""Tests for the observation policy module.

Covers:
- Threshold routing (full / compressed / metadata_only / drop)
- D4 safety override (FAILURE status, ERROR/RETRY/FINAL_RESPONSE types always FULL)
- COMPRESSED byte-truncation
- METADATA_ONLY nulling of input/output
- DROP skeleton retention
- Fail-open in collector
- Determinism (same inputs, same outputs)
"""

from __future__ import annotations

import uuid
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    ObservationLevel,
    create_event,
)
from agentlens.observation import (
    ImportanceScorer,
    ObservationPolicy,
    PolicyConfig,
    ScoringConfig,
    build_policy_and_scorer,
)
from agentlens.observation.policy import PolicyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACE_ID = uuid.uuid4()


def _make_event(
    event_type: EventType = EventType.LLM_CALL,
    status: EventStatus = EventStatus.SUCCESS,
    parent_id: uuid.UUID | None = uuid.uuid4(),
    latency_ms: float | None = None,
    metadata: dict | None = None,
    input_data: dict | None = None,
    output_data: dict | None = None,
) -> AgentEvent:
    return create_event(
        trace_id=TRACE_ID,
        agent_id="test-agent",
        event_type=event_type,
        status=status,
        parent_id=parent_id,
        latency_ms=latency_ms,
        metadata=metadata,
        input_data=input_data,
        output_data=output_data,
    )


@pytest.fixture
def scorer() -> ImportanceScorer:
    return ImportanceScorer()


@pytest.fixture
def adaptive_policy() -> ObservationPolicy:
    return ObservationPolicy(PolicyConfig(mode="adaptive"))


@pytest.fixture
def full_policy() -> ObservationPolicy:
    return ObservationPolicy(PolicyConfig(mode="full"))


# ---------------------------------------------------------------------------
# 1. Full mode is a no-op
# ---------------------------------------------------------------------------

class TestFullMode:
    def test_full_mode_always_returns_full(self, full_policy):
        ev = _make_event(status=EventStatus.SUCCESS)
        assert full_policy.decide(0.0, ev) == ObservationLevel.FULL

    def test_full_mode_transform_unchanged(self, full_policy):
        ev = _make_event(input_data={"x": 1})
        result = full_policy.transform(ev, ObservationLevel.FULL)
        assert result.observation_level == ObservationLevel.FULL
        assert result.input_data == {"x": 1}


# ---------------------------------------------------------------------------
# 2. Threshold routing
# ---------------------------------------------------------------------------

class TestThresholdRouting:
    """PolicyConfig defaults: compress=0.60, metadata=0.35, drop=0.15."""

    @pytest.mark.parametrize("score,expected", [
        (0.90, ObservationLevel.FULL),
        (0.60, ObservationLevel.FULL),         # boundary: >= compress -> FULL
        (0.59, ObservationLevel.COMPRESSED),
        (0.35, ObservationLevel.COMPRESSED),   # boundary: >= metadata -> COMP
        (0.34, ObservationLevel.METADATA_ONLY),
        (0.15, ObservationLevel.METADATA_ONLY),# boundary: >= drop -> META
        (0.14, ObservationLevel.DROP),
        (0.00, ObservationLevel.DROP),
    ])
    def test_routing(self, adaptive_policy, score, expected):
        ev = _make_event(event_type=EventType.LLM_CALL, status=EventStatus.SUCCESS)
        assert adaptive_policy.decide(score, ev) == expected


# ---------------------------------------------------------------------------
# 3. D4 safety override
# ---------------------------------------------------------------------------

class TestD4SafetyOverride:
    def test_failure_status_always_full(self, adaptive_policy):
        ev = _make_event(status=EventStatus.FAILURE)
        assert adaptive_policy.decide(0.0, ev) == ObservationLevel.FULL

    @pytest.mark.parametrize("etype", [
        EventType.ERROR, EventType.RETRY, EventType.FINAL_RESPONSE
    ])
    def test_always_full_types(self, adaptive_policy, etype):
        ev = _make_event(event_type=etype, status=EventStatus.SUCCESS)
        # Even with a very low score, these event types must be FULL.
        assert adaptive_policy.decide(0.0, ev) == ObservationLevel.FULL


# ---------------------------------------------------------------------------
# 4. COMPRESSED transform — byte truncation
# ---------------------------------------------------------------------------

class TestCompressedTransform:
    def test_small_payload_unchanged(self, adaptive_policy):
        ev = _make_event(input_data={"key": "value"}, output_data={"result": "ok"})
        result = adaptive_policy.transform(ev, ObservationLevel.COMPRESSED)
        assert result.observation_level == ObservationLevel.COMPRESSED
        assert result.input_data == {"key": "value"}
        assert result.output_data == {"result": "ok"}

    def test_large_payload_truncated(self):
        policy = ObservationPolicy(PolicyConfig(mode="adaptive", payload_max_bytes=50))
        big_data = {"data": "x" * 200}
        ev = _make_event(input_data=big_data)
        result = policy.transform(ev, ObservationLevel.COMPRESSED)
        assert result.input_data is not None
        assert result.input_data.get("_truncated") is True

    def test_none_payload_stays_none(self, adaptive_policy):
        ev = _make_event(input_data=None, output_data=None)
        result = adaptive_policy.transform(ev, ObservationLevel.COMPRESSED)
        assert result.input_data is None
        assert result.output_data is None


# ---------------------------------------------------------------------------
# 5. METADATA_ONLY transform
# ---------------------------------------------------------------------------

class TestMetadataOnlyTransform:
    def test_nulls_input_and_output(self, adaptive_policy):
        ev = _make_event(input_data={"prompt": "hello"}, output_data={"response": "world"})
        result = adaptive_policy.transform(ev, ObservationLevel.METADATA_ONLY)
        assert result.observation_level == ObservationLevel.METADATA_ONLY
        assert result.input_data is None
        assert result.output_data is None

    def test_preserves_identity_fields(self, adaptive_policy):
        ev = _make_event(input_data={"prompt": "hello"})
        result = adaptive_policy.transform(ev, ObservationLevel.METADATA_ONLY)
        assert result.event_id == ev.event_id
        assert result.trace_id == ev.trace_id
        assert result.event_type == ev.event_type


# ---------------------------------------------------------------------------
# 6. DROP skeleton
# ---------------------------------------------------------------------------

class TestDropSkeleton:
    def test_drop_retains_skeleton_by_default(self, adaptive_policy):
        ev = _make_event(
            input_data={"secret": "data"},
            output_data={"answer": "42"},
            metadata={"step": "final"},
        )
        result = adaptive_policy.transform(ev, ObservationLevel.DROP)
        assert result.observation_level == ObservationLevel.DROP
        assert result.input_data is None
        assert result.output_data is None
        assert result.metadata == {"_dropped": True}
        # Identity preserved
        assert result.event_id == ev.event_id
        assert result.trace_id == ev.trace_id
        assert result.event_type == ev.event_type
        assert result.status == ev.status

    def test_drop_no_skeleton(self):
        policy = ObservationPolicy(PolicyConfig(mode="adaptive", drop_retain_skeleton=False))
        ev = _make_event(input_data={"x": 1}, metadata={"step": "s"})
        result = policy.transform(ev, ObservationLevel.DROP)
        assert result.metadata is None


# ---------------------------------------------------------------------------
# 7. ImportanceScorer determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_score(self, scorer):
        ev = _make_event(latency_ms=500.0)
        s1 = scorer.score(ev, repeat_index=0)
        s2 = scorer.score(ev, repeat_index=0)
        assert s1 == s2

    def test_score_clamped(self, scorer):
        ev = _make_event(
            event_type=EventType.ERROR,
            status=EventStatus.FAILURE,
            parent_id=None,
            latency_ms=2000.0,
            metadata={"exception_type": "RuntimeError"},
        )
        score = scorer.score(ev, repeat_index=0)
        assert 0.0 <= score <= 1.0

    def test_repetition_lowers_score(self, scorer):
        ev = _make_event()
        s0 = scorer.score(ev, repeat_index=0)
        s3 = scorer.score(ev, repeat_index=3)
        assert s3 < s0

    def test_failure_scores_higher_than_success(self, scorer):
        ev_fail = _make_event(status=EventStatus.FAILURE)
        ev_ok   = _make_event(status=EventStatus.SUCCESS)
        assert scorer.score(ev_fail) > scorer.score(ev_ok)


# ---------------------------------------------------------------------------
# 8. Fail-open in collector
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collector_fail_open_on_policy_error():
    """If the scorer raises, the event is still saved unmodified (D7)."""
    from agentlens.sdk_collector.collector import EventCollector

    mock_storage = AsyncMock()
    mock_storage.save = AsyncMock()

    bad_scorer = MagicMock()
    bad_scorer.score.side_effect = RuntimeError("scorer exploded")

    policy = ObservationPolicy(PolicyConfig(mode="adaptive"))

    collector = EventCollector(storage_backend=mock_storage, scorer=bad_scorer, policy=policy)
    await collector.start_trace(agent_id="test")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev = await collector.record_event(
            event_type=EventType.LLM_CALL,
            input_data={"x": 1},
        )

    await collector.end_trace()

    # Storage was still called (fail-open).
    mock_storage.save.assert_called_once()
    # A RuntimeWarning was emitted.
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


# ---------------------------------------------------------------------------
# 9. build_policy_and_scorer factory
# ---------------------------------------------------------------------------

def test_factory_full_mode_is_noop(monkeypatch):
    monkeypatch.setenv("AGENTLENS_OBSERVATION_POLICY", "full")
    scorer, policy = build_policy_and_scorer()
    ev = _make_event()
    assert policy.decide(0.0, ev) == ObservationLevel.FULL


def test_factory_adaptive_mode(monkeypatch):
    monkeypatch.setenv("AGENTLENS_OBSERVATION_POLICY", "adaptive")
    scorer, policy = build_policy_and_scorer()
    ev = _make_event(event_type=EventType.LLM_CALL, status=EventStatus.SUCCESS)
    # Score will be computed; routing will produce something other than FULL for low score.
    # Just check that the factory wires everything without errors.
    score = scorer.score(ev)
    level = policy.decide(score, ev)
    assert isinstance(level, ObservationLevel)
