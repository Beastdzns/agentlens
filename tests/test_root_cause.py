"""
Unit tests for the Phase 11 root-cause analyzer.

Every analyzer rule has a test: the primary/secondary split, the root-cause
selection and its tie-breaks, the recovered verdict, the suspected-upstream
fallback, the category derivation, the confidence ladder and the D5 degraded
payload behaviour.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from agentlens.causal.analyzer import (
    CATEGORY_INVALID_INPUT,
    CATEGORY_LLM_ERROR,
    CATEGORY_NONE,
    CATEGORY_SUSPECTED_UPSTREAM,
    CATEGORY_TIMEOUT,
    CATEGORY_TOOL_ERROR,
    CATEGORY_UNKNOWN,
    VERDICT_INSUFFICIENT_EVIDENCE,
    VERDICT_NO_FAILURE,
    VERDICT_RECOVERED,
    VERDICT_ROOT_CAUSE,
    AnalyzerConfig,
    RootCauseAnalyzer,
)
from agentlens.causal.explain import NO_DETAIL_PHRASE, explain
from agentlens.causal.graph import build_graph
from agentlens.events import AgentEvent, EventStatus, EventType, ObservationLevel

pytestmark = pytest.mark.unit

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(
    *,
    event_type: EventType = EventType.LLM_CALL,
    status: EventStatus = EventStatus.SUCCESS,
    parent_id=None,
    metadata=None,
    input_data=None,
    output_data=None,
    timestamp=None,
    importance_score: float = 0.5,
    observation_level: ObservationLevel = ObservationLevel.FULL,
    trace_id=None,
) -> AgentEvent:
    return AgentEvent(
        trace_id=trace_id or uuid4(),
        agent_id="test-agent",
        event_type=event_type,
        status=status,
        parent_id=parent_id,
        metadata=metadata,
        input_data=input_data,
        output_data=output_data,
        importance_score=importance_score,
        observation_level=observation_level,
        **({"timestamp": timestamp} if timestamp is not None else {}),
    )


def _analyze(events):
    return RootCauseAnalyzer().analyze(build_graph(events))


# ---------------------------------------------------------------------------
# No failure
# ---------------------------------------------------------------------------


def test_trace_without_failures_reports_no_failure() -> None:
    events = [
        _event(event_type=EventType.LLM_CALL),
        _event(event_type=EventType.FINAL_RESPONSE),
    ]
    result = _analyze(events)

    assert result.verdict == VERDICT_NO_FAILURE
    assert result.root_cause_event_id is None
    assert result.confidence == 0.0
    assert result.category == CATEGORY_NONE
    assert result.primary_failures == []
    assert result.secondary_failures == []
    assert result.affected_events == []
    assert result.recovered_failures == []


def test_empty_trace_reports_no_failure() -> None:
    result = _analyze([])
    assert result.verdict == VERDICT_NO_FAILURE
    assert result.root_cause_event_id is None


def test_no_failure_explanation_mentions_no_failure() -> None:
    result = _analyze([_event()])
    assert "no failure" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Primary vs. secondary failures
# ---------------------------------------------------------------------------


def test_failure_without_failed_ancestor_is_primary() -> None:
    root = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    result = _analyze([root])

    assert result.primary_failures == [str(root.event_id)]
    assert result.secondary_failures == []


def test_failure_with_failed_ancestor_is_secondary() -> None:
    root = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    child = _event(
        status=EventStatus.FAILURE,
        parent_id=root.event_id,
        output_data={"error": "downstream"},
    )
    result = _analyze([root, child])

    assert result.primary_failures == [str(root.event_id)]
    assert result.secondary_failures == [str(child.event_id)]


def test_error_event_type_counts_as_failure() -> None:
    error = _event(event_type=EventType.ERROR, status=EventStatus.SUCCESS)
    result = _analyze([error])

    assert result.verdict == VERDICT_ROOT_CAUSE
    assert result.root_cause_event_id == str(error.event_id)


# ---------------------------------------------------------------------------
# Root-cause selection
# ---------------------------------------------------------------------------


def test_ancestor_most_failure_is_the_root_cause() -> None:
    root = _event(status=EventStatus.FAILURE, output_data={"error": "root"})
    mid = _event(status=EventStatus.FAILURE, parent_id=root.event_id)
    leaf = _event(status=EventStatus.FAILURE, parent_id=mid.event_id)

    result = _analyze([root, mid, leaf])

    assert result.root_cause_event_id == str(root.event_id)
    assert result.verdict == VERDICT_ROOT_CAUSE
    assert result.evidence_path == [
        str(root.event_id),
        str(mid.event_id),
        str(leaf.event_id),
    ]


def test_candidate_with_most_descendants_wins() -> None:
    wide = _event(status=EventStatus.FAILURE, output_data={"error": "wide"})
    wide_children = [
        _event(parent_id=wide.event_id) for _ in range(3)
    ]
    narrow = _event(status=EventStatus.FAILURE, output_data={"error": "narrow"})

    result = _analyze([wide, *wide_children, narrow])

    assert result.root_cause_event_id == str(wide.event_id)
    assert len(result.affected_events) == 3


def test_selection_is_deterministic_for_equal_candidates() -> None:
    first = _event(
        status=EventStatus.FAILURE,
        output_data={"error": "first"},
        timestamp=_BASE,
    )
    second = _event(
        status=EventStatus.FAILURE,
        output_data={"error": "second"},
        timestamp=_BASE + timedelta(seconds=1),
    )

    result = _analyze([second, first])

    # Same descendant count and depth -> the earlier timestamp wins.
    assert result.root_cause_event_id == str(first.event_id)


def test_affected_events_exclude_failures() -> None:
    root = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    child = _event(parent_id=root.event_id)
    failing_child = _event(status=EventStatus.FAILURE, parent_id=root.event_id)

    result = _analyze([root, child, failing_child])

    assert result.affected_events == [str(child.event_id)]


# ---------------------------------------------------------------------------
# Recovered verdict
# ---------------------------------------------------------------------------


def test_failed_attempt_followed_by_successful_retry_is_recovered() -> None:
    attempt = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    retry = _event(metadata={"retry_of": str(attempt.event_id)})

    result = _analyze([attempt, retry])

    assert result.verdict == VERDICT_RECOVERED
    assert result.root_cause_event_id is None
    assert result.recovered_failures == [str(attempt.event_id)]


def test_recovered_chain_through_two_attempts() -> None:
    first = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    second = _event(status=EventStatus.FAILURE, metadata={"retry_of": str(first.event_id)})
    third = _event(metadata={"retry_of": str(second.event_id)})

    result = _analyze([first, second, third])

    assert result.verdict == VERDICT_RECOVERED
    assert set(result.recovered_failures) == {str(first.event_id), str(second.event_id)}


def test_unrecovered_failure_alongside_a_recovered_one_is_reported() -> None:
    recovered = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    retry = _event(metadata={"retry_of": str(recovered.event_id)})
    fatal = _event(status=EventStatus.FAILURE, output_data={"error": "fatal"})

    result = _analyze([recovered, retry, fatal])

    assert result.verdict == VERDICT_ROOT_CAUSE
    assert result.root_cause_event_id == str(fatal.event_id)
    assert result.recovered_failures == [str(recovered.event_id)]


# ---------------------------------------------------------------------------
# Suspicious upstream
# ---------------------------------------------------------------------------


def test_successful_upstream_with_wrong_output_is_suspected() -> None:
    retrieval = _event(
        event_type=EventType.RETRIEVAL,
        output_data={"error": "no relevant documents"},
    )
    decision = _event(
        event_type=EventType.DECISION,
        status=EventStatus.FAILURE,
        parent_id=retrieval.event_id,
        output_data={"error": "cannot decide"},
    )

    result = _analyze([retrieval, decision])

    assert result.verdict == VERDICT_INSUFFICIENT_EVIDENCE
    assert result.root_cause_event_id == str(retrieval.event_id)
    assert result.category == CATEGORY_SUSPECTED_UPSTREAM
    assert result.confidence == 0.5


def test_exception_type_also_marks_an_upstream_as_suspicious() -> None:
    upstream = _event(metadata={"exception_type": "RuntimeError"})
    failure = _event(
        status=EventStatus.FAILURE,
        parent_id=upstream.event_id,
        output_data={"error": "boom"},
    )

    result = _analyze([upstream, failure])

    assert result.category == CATEGORY_SUSPECTED_UPSTREAM
    assert result.root_cause_event_id == str(upstream.event_id)


def test_clean_upstream_leaves_the_failure_as_the_root_cause() -> None:
    upstream = _event(output_data={"result": "fine"})
    failure = _event(
        status=EventStatus.FAILURE,
        parent_id=upstream.event_id,
        output_data={"error": "boom"},
    )

    result = _analyze([upstream, failure])

    assert result.verdict == VERDICT_ROOT_CAUSE
    assert result.root_cause_event_id == str(failure.event_id)
    assert result.confidence == 0.3


def test_failed_ancestor_takes_precedence_over_suspicious_upstream() -> None:
    upstream = _event(output_data={"error": "suspicious"})
    failure = _event(
        status=EventStatus.FAILURE,
        parent_id=upstream.event_id,
        output_data={"error": "boom"},
    )
    downstream = _event(status=EventStatus.FAILURE, parent_id=failure.event_id)

    result = _analyze([upstream, failure, downstream])

    assert result.verdict == VERDICT_ROOT_CAUSE
    assert result.root_cause_event_id == str(failure.event_id)
    assert result.category != CATEGORY_SUSPECTED_UPSTREAM


# ---------------------------------------------------------------------------
# Category derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_message", "expected"),
    [
        ("Request timeout after 30s", CATEGORY_TIMEOUT),
        ("the request timed out", CATEGORY_TIMEOUT),
        ("invalid request payload", CATEGORY_INVALID_INPUT),
        ("bad request", CATEGORY_INVALID_INPUT),
        ("HTTP 400 response", CATEGORY_INVALID_INPUT),
    ],
)
def test_categories_from_the_error_message(error_message, expected) -> None:
    failure = _event(status=EventStatus.FAILURE, output_data={"error": error_message})
    result = _analyze([failure])
    assert result.category == expected


def test_tool_error_category() -> None:
    failure = _event(
        event_type=EventType.TOOL_CALL,
        status=EventStatus.FAILURE,
        output_data={"error": "tool exploded"},
    )
    result = _analyze([failure])
    assert result.category == CATEGORY_TOOL_ERROR


def test_llm_error_category() -> None:
    failure = _event(
        event_type=EventType.LLM_CALL,
        status=EventStatus.FAILURE,
        output_data={"error": "model exploded"},
    )
    result = _analyze([failure])
    assert result.category == CATEGORY_LLM_ERROR


def test_unknown_category() -> None:
    failure = _event(
        event_type=EventType.RETRIEVAL,
        status=EventStatus.FAILURE,
        output_data={"error": "something odd"},
    )
    result = _analyze([failure])
    assert result.category == CATEGORY_UNKNOWN


# ---------------------------------------------------------------------------
# Confidence ladder
# ---------------------------------------------------------------------------


def test_confidence_is_one_when_the_chain_has_an_exception_type() -> None:
    root = _event(
        status=EventStatus.FAILURE,
        output_data={"error": "boom"},
        metadata={"exception_type": "RuntimeError"},
    )
    leaf = _event(status=EventStatus.FAILURE, parent_id=root.event_id)

    result = _analyze([root, leaf])

    assert result.evidence_path == [str(root.event_id), str(leaf.event_id)]
    assert result.confidence == 1.0


def test_confidence_is_eight_tenths_for_a_chain_without_exception_type() -> None:
    root = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    leaf = _event(status=EventStatus.FAILURE, parent_id=root.event_id)

    result = _analyze([root, leaf])

    assert result.confidence == 0.8


def test_confidence_is_three_tenths_without_causal_context() -> None:
    failure = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    result = _analyze([failure])
    assert result.confidence == 0.3


# ---------------------------------------------------------------------------
# D5 — degraded payloads
# ---------------------------------------------------------------------------


def test_dropped_payload_caps_the_confidence() -> None:
    root = _event(
        status=EventStatus.FAILURE,
        observation_level=ObservationLevel.DROP,
        output_data=None,
    )
    leaf = _event(status=EventStatus.FAILURE, parent_id=root.event_id)

    result = _analyze([root, leaf])

    assert result.confidence <= 0.5


def test_dropped_payload_explains_that_no_detail_was_retained() -> None:
    failure = _event(
        status=EventStatus.FAILURE,
        observation_level=ObservationLevel.DROP,
        output_data=None,
    )
    result = _analyze([failure])

    assert NO_DETAIL_PHRASE in result.explanation.lower()
    assert result.root_cause_event_id == str(failure.event_id)


def test_metadata_only_payload_is_handled_gracefully() -> None:
    failure = _event(
        status=EventStatus.FAILURE,
        observation_level=ObservationLevel.METADATA_ONLY,
        output_data=None,
    )
    result = _analyze([failure])

    assert result.verdict == VERDICT_ROOT_CAUSE
    assert result.confidence <= 0.5


# ---------------------------------------------------------------------------
# Grounded explanation
# ---------------------------------------------------------------------------


def test_explanation_cites_the_recorded_error_message() -> None:
    failure = _event(
        status=EventStatus.FAILURE,
        output_data={"error": "the disk was full"},
    )
    result = _analyze([failure])

    assert "the disk was full" in result.explanation


def test_explanation_contains_no_invented_cause() -> None:
    failure = _event(
        status=EventStatus.FAILURE,
        output_data={"error": "boom"},
    )
    result = _analyze([failure])

    lowered = result.explanation.lower()
    for forbidden in ("probably", "likely because", "the model was confused"):
        assert forbidden not in lowered


def test_explain_is_deterministic() -> None:
    failure = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    graph = build_graph([failure])
    result = RootCauseAnalyzer().analyze(graph)

    assert explain(result, graph) == explain(result, graph)
    assert explain(result, graph) == result.explanation


def test_explanation_of_a_chain_mentions_the_evidence_path() -> None:
    root = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    leaf = _event(status=EventStatus.FAILURE, parent_id=root.event_id)

    result = _analyze([root, leaf])

    assert "evidence path" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Determinism & configuration
# ---------------------------------------------------------------------------


def test_analysis_is_deterministic_for_shuffled_input() -> None:
    root = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    child = _event(status=EventStatus.FAILURE, parent_id=root.event_id)

    forward = _analyze([root, child])
    backward = _analyze([child, root])

    assert forward.to_dict() == backward.to_dict()


def test_analyzer_accepts_a_custom_config() -> None:
    analyzer = RootCauseAnalyzer(AnalyzerConfig(failure_statuses=frozenset({"failure"})))
    failure = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    result = analyzer.analyze(build_graph([failure]))
    assert result.root_cause_event_id == str(failure.event_id)


def test_result_is_json_serialisable() -> None:
    import json

    failure = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    result = _analyze([failure])
    json.dumps(result.to_dict())


def test_analysis_of_a_large_trace_is_fast() -> None:
    """NFR1: analysis of 1,000 events must stay well under 100 ms."""
    import time

    root = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    events = [root]
    for _ in range(999):
        events.append(_event(parent_id=root.event_id))

    start = time.perf_counter()
    result = _analyze(events)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result.root_cause_event_id == str(root.event_id)
    assert elapsed_ms < 100, f"analysis took {elapsed_ms:.1f} ms"
