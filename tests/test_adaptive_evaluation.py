"""
Evaluation invariants for adaptive observability (Phase 10).

These tests pin the *guarantees* that the adaptive pipeline must uphold, so
that a future change to the scoring rules or the observation policy cannot
silently degrade diagnosis quality.

Invariants:
- Failures are always observed in full (D4) and keep their exception evidence.
- Dropped events keep their identity and structure, only the payload goes.
- Repetition lowers the score of repeated *successful* steps.
- Full mode is a strict no-op.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from agentlens.events import AgentEvent, EventStatus, EventType, ObservationLevel
from agentlens.observation import (
    ImportanceScorer,
    ObservationPolicy,
    PolicyConfig,
    ScoringConfig,
)


@pytest.fixture
def base_event():
    """A normal, successful LLM call with a payload worth keeping."""
    return AgentEvent(
        event_id=str(uuid4()),
        trace_id=str(uuid4()),
        parent_id=str(uuid4()),
        agent_id="test-agent",
        event_type=EventType.LLM_CALL,
        status=EventStatus.SUCCESS,
        timestamp=datetime.now(timezone.utc),
        input_data={"prompt": "test " * 100},
        output_data={"response": "result " * 100},
        latency_ms=150.0,
        metadata={},
    )


def test_failure_is_always_full(base_event):
    """D4: a failure is observed in full even when its score is 0."""
    policy = ObservationPolicy(PolicyConfig(mode="adaptive"))
    event = base_event.model_copy(
        update={"status": EventStatus.FAILURE, "event_type": EventType.LLM_CALL}
    )

    level = policy.decide(0.0, event)  # Force score to 0
    assert level == ObservationLevel.FULL

    transformed = policy.transform(event, level)
    assert transformed.input_data == event.input_data
    assert transformed.output_data == event.output_data


def test_failure_keeps_exception_evidence(base_event):
    """The exception type/message must survive observation for diagnosis."""
    policy = ObservationPolicy(PolicyConfig(mode="adaptive"))
    event = base_event.model_copy(
        update={
            "status": EventStatus.FAILURE,
            "metadata": {"exception_type": "TimeoutError", "exception_message": "boom"},
        }
    )

    transformed = policy.transform(event, policy.decide(0.0, event))
    assert transformed.metadata["exception_type"] == "TimeoutError"
    assert transformed.metadata["exception_message"] == "boom"


def test_drop_retains_skeleton(base_event):
    """D5: a dropped event keeps its identity and structure, not its payload."""
    policy = ObservationPolicy(PolicyConfig(mode="adaptive", drop_retain_skeleton=True))
    level = ObservationLevel.DROP

    transformed = policy.transform(base_event, level)

    # Structure remains
    assert transformed.event_id == base_event.event_id
    assert transformed.trace_id == base_event.trace_id
    assert transformed.parent_id == base_event.parent_id
    assert transformed.status == base_event.status
    assert transformed.event_type == base_event.event_type
    assert transformed.latency_ms == base_event.latency_ms

    # Payloads wiped
    assert transformed.input_data is None
    assert transformed.output_data is None
    assert transformed.metadata == {"_dropped": True}


def test_repetition_penalty_drops_score(base_event):
    """A repeated successful step loses importance (bounded by the cap)."""
    scorer = ImportanceScorer(ScoringConfig(repetition_penalty=0.05, repetition_cap=3))
    event = base_event.model_copy(
        update={"event_type": EventType.LLM_CALL, "status": EventStatus.SUCCESS}
    )

    score_first = scorer.score(event, repeat_index=0)
    score_third = scorer.score(event, repeat_index=3)

    assert score_third < score_first
    assert score_first - score_third == pytest.approx(0.15)  # 3 * 0.05


def test_repetition_penalty_skips_failures(base_event):
    """A failure never loses score to repetition — it must stay diagnosable."""
    scorer = ImportanceScorer()
    event = base_event.model_copy(update={"status": EventStatus.FAILURE})
    assert scorer.score(event, repeat_index=5) == scorer.score(event, repeat_index=0)


def test_repetition_penalty_skips_decisions(base_event):
    """Decisions are exempt from the repetition penalty."""
    scorer = ImportanceScorer()
    event = base_event.model_copy(update={"event_type": EventType.DECISION})
    assert scorer.score(event, repeat_index=5) == scorer.score(event, repeat_index=0)


def test_full_mode_no_op(base_event):
    """Full mode must not change the event in any way."""
    policy = ObservationPolicy(PolicyConfig(mode="full"))
    level = policy.decide(0.0, base_event)
    assert level == ObservationLevel.FULL
    transformed = policy.transform(base_event, level)
    assert transformed.input_data == base_event.input_data
    assert transformed.output_data == base_event.output_data
    assert transformed.metadata == base_event.metadata
