"""
Labelled causal scenarios — the evaluation set for Phase 11 (member N).

Each scenario is a small, fully labelled trace: the events, the expected
verdict and the expected root cause.  The labels are the *ground truth* the
causal analyzer is measured against, and the same set is used by

- ``tests/test_causal_evaluation.py`` — correctness per scenario;
- ``tests/benchmark_causal.py`` — the H3 comparison against a chronological
  baseline, latency, robustness and explanation quality.

The event ids are fixed (not random) so the labels stay stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from agentlens.events import AgentEvent, EventStatus, EventType, ObservationLevel

#: Every scenario belongs to the same trace.
TRACE_ID = UUID("11111111-1111-4111-8111-111111111111")

#: Base timestamp of the scenarios.
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def event_id(index: int) -> str:
    """A deterministic event id for a scenario."""
    return f"22222222-2222-4222-8222-{index:012d}"


@dataclass(frozen=True)
class CausalScenario:
    """A labelled trace plus the expected diagnosis."""

    id: str
    description: str
    events: tuple[AgentEvent, ...]
    expected_verdict: str
    expected_root_cause: Optional[str]
    expected_category: str
    expected_primary_failures: tuple[str, ...] = ()
    expected_secondary_failures: tuple[str, ...] = ()
    expected_affected_events: tuple[str, ...] = ()
    expected_confidence: float = 0.0
    #: The recorded error message the explanation has to quote (if any).
    expected_error_message: Optional[str] = None
    #: Whether the scenario can be answered without the failed step's payload.
    #: ``False`` for scenarios whose evidence *is* the payload.
    survives_payload_drop: bool = True
    notes: str = ""

    def event(self, index: int) -> AgentEvent:
        """Return the event with the given index."""
        return self.events[index]

    @property
    def root_cause_event(self) -> Optional[AgentEvent]:
        """Return the labelled root-cause event, if any."""
        if self.expected_root_cause is None:
            return None
        for event in self.events:
            if str(event.event_id) == self.expected_root_cause:
                return event
        return None


def _event(
    index: int,
    *,
    event_type: EventType,
    status: EventStatus = EventStatus.SUCCESS,
    parent: Optional[int] = None,
    seconds: int = 0,
    input_data: Optional[dict] = None,
    output_data: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> AgentEvent:
    return AgentEvent(
        event_id=UUID(event_id(index)),
        trace_id=TRACE_ID,
        agent_id="scenario-agent",
        event_type=event_type,
        status=status,
        parent_id=UUID(event_id(parent)) if parent is not None else None,
        timestamp=T0 + timedelta(seconds=seconds),
        input_data=input_data,
        output_data=output_data,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------

_TOOL_ERROR = CausalScenario(
    id="tool_error",
    description="A tool call fails and the final answer fails with it.",
    events=(
        _event(
            1,
            event_type=EventType.TOOL_CALL,
            status=EventStatus.FAILURE,
            seconds=1,
            output_data={"error": "search backend unavailable"},
            metadata={"exception_type": "RuntimeError"},
        ),
        _event(
            2,
            event_type=EventType.FINAL_RESPONSE,
            status=EventStatus.FAILURE,
            parent=1,
            seconds=2,
            output_data={"error": "no answer could be produced"},
        ),
    ),
    expected_verdict="root_cause",
    expected_root_cause=event_id(1),
    expected_category="tool_error",
    expected_primary_failures=(event_id(1),),
    expected_secondary_failures=(event_id(2),),
    expected_confidence=1.0,
    expected_error_message="search backend unavailable",
    notes="The classic case: the ancestor-most failure is the root cause.",
)

_TIMEOUT = CausalScenario(
    id="timeout",
    description="An LLM call times out and the agent aborts.",
    events=(
        _event(
            1,
            event_type=EventType.LLM_CALL,
            status=EventStatus.FAILURE,
            seconds=1,
            output_data={"error": "Request timeout after 30 seconds"},
            metadata={"exception_type": "TimeoutError"},
        ),
        _event(
            2,
            event_type=EventType.ERROR,
            status=EventStatus.FAILURE,
            parent=1,
            seconds=2,
            output_data={"error": "agent aborted"},
        ),
    ),
    expected_verdict="root_cause",
    expected_root_cause=event_id(1),
    expected_category="timeout",
    expected_primary_failures=(event_id(1),),
    expected_secondary_failures=(event_id(2),),
    expected_confidence=1.0,
    expected_error_message="Request timeout after 30 seconds",
    notes="The category comes from the recorded error message alone.",
)

_INVALID_INPUT = CausalScenario(
    id="invalid_input",
    description="The first step fails on bad input; nothing downstream fails.",
    events=(
        _event(
            1,
            event_type=EventType.LLM_CALL,
            status=EventStatus.FAILURE,
            seconds=1,
            output_data={"error": "invalid request: query must not be blank"},
        ),
    ),
    expected_verdict="root_cause",
    expected_root_cause=event_id(1),
    expected_category="invalid_input",
    expected_primary_failures=(event_id(1),),
    expected_secondary_failures=(),
    expected_confidence=0.3,
    expected_error_message="invalid request: query must not be blank",
    notes="No cascade, so the confidence stays low (no causal context).",
)

_RETRY_RECOVERED = CausalScenario(
    id="retry_recovered",
    description="The first attempt fails, the retry succeeds.",
    events=(
        _event(
            1,
            event_type=EventType.TOOL_CALL,
            status=EventStatus.FAILURE,
            seconds=1,
            output_data={"error": "transient connection reset"},
        ),
        _event(
            2,
            event_type=EventType.RETRY,
            status=EventStatus.SUCCESS,
            seconds=2,
            output_data={"result": "ok"},
            metadata={"retry_of": event_id(1), "attempt": 2},
        ),
    ),
    expected_verdict="recovered",
    expected_root_cause=None,
    expected_category="none",
    expected_primary_failures=(),
    expected_secondary_failures=(),
    expected_confidence=0.0,
    expected_error_message="transient connection reset",
    notes="A recovered failure is reported, but it is not a root cause.",
)

_BAD_UPSTREAM = CausalScenario(
    id="bad_upstream",
    description="A retrieval returns wrong data, the decision built on it fails.",
    events=(
        _event(
            1,
            event_type=EventType.RETRIEVAL,
            status=EventStatus.SUCCESS,
            seconds=1,
            output_data={"error": "no relevant documents found"},
        ),
        _event(
            2,
            event_type=EventType.DECISION,
            status=EventStatus.FAILURE,
            parent=1,
            seconds=2,
            output_data={"error": "cannot decide without data"},
        ),
        _event(
            3,
            event_type=EventType.FINAL_RESPONSE,
            status=EventStatus.SUCCESS,
            parent=2,
            seconds=3,
            output_data={"answer": "a wrong answer"},
        ),
    ),
    expected_verdict="insufficient_evidence",
    expected_root_cause=event_id(1),
    expected_category="suspected_upstream",
    expected_primary_failures=(event_id(2),),
    expected_secondary_failures=(),
    expected_affected_events=(event_id(3),),
    expected_confidence=0.5,
    expected_error_message="no relevant documents found",
    survives_payload_drop=False,
    notes=(
        "The upstream step succeeded but its output reports a problem — the "
        "analyzer reports it as a suspected upstream cause, not a confirmed one."
    ),
)

_CASCADE = CausalScenario(
    id="cascade",
    description="A retrieval fails, the decision fails, the answer fails.",
    events=(
        _event(
            1,
            event_type=EventType.RETRIEVAL,
            status=EventStatus.FAILURE,
            seconds=1,
            output_data={"error": "index unavailable"},
        ),
        _event(
            2,
            event_type=EventType.DECISION,
            status=EventStatus.FAILURE,
            parent=1,
            seconds=2,
            output_data={"error": "no data to decide on"},
        ),
        _event(
            3,
            event_type=EventType.FINAL_RESPONSE,
            status=EventStatus.FAILURE,
            parent=2,
            seconds=3,
            output_data={"error": "no answer"},
        ),
    ),
    expected_verdict="root_cause",
    expected_root_cause=event_id(1),
    expected_category="unknown",
    expected_primary_failures=(event_id(1),),
    expected_secondary_failures=(event_id(2), event_id(3)),
    expected_confidence=0.8,
    expected_error_message="index unavailable",
    notes="A three-step cascade — only the ancestor-most failure is the cause.",
)

_SKEWED_CLOCK = CausalScenario(
    id="skewed_clock",
    description=(
        "The root cause is recorded with a later timestamp than its child "
        "(skewed clock) — the graph, not the clock, is authoritative."
    ),
    events=(
        _event(
            1,
            event_type=EventType.TOOL_CALL,
            status=EventStatus.FAILURE,
            seconds=10,
            output_data={"error": "upstream service returned garbage"},
        ),
        _event(
            2,
            event_type=EventType.FINAL_RESPONSE,
            status=EventStatus.FAILURE,
            parent=1,
            seconds=0,
            output_data={"error": "no answer"},
        ),
    ),
    expected_verdict="root_cause",
    expected_root_cause=event_id(1),
    expected_category="tool_error",
    expected_primary_failures=(event_id(1),),
    expected_secondary_failures=(event_id(2),),
    expected_confidence=0.8,
    expected_error_message="upstream service returned garbage",
    notes=(
        "A chronological baseline picks the wrong event here; the graph does "
        "not."
    ),
)


#: Every scenario of the evaluation set.
SCENARIOS: tuple[CausalScenario, ...] = (
    _TOOL_ERROR,
    _TIMEOUT,
    _INVALID_INPUT,
    _RETRY_RECOVERED,
    _BAD_UPSTREAM,
    _CASCADE,
    _SKEWED_CLOCK,
)


def scenario_by_id(scenario_id: str) -> CausalScenario:
    """Return the scenario with the given id."""
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"unknown scenario: {scenario_id}")


# ---------------------------------------------------------------------------
# Helpers used by the evaluation and the benchmark
# ---------------------------------------------------------------------------


def baseline_first_failure(events) -> Optional[str]:
    """
    The naive baseline: "the first failure, by wall-clock".

    This is what an observer without a causal graph can do — sort the events
    by timestamp and take the first one that failed.  It is the reference the
    graph-based localization is measured against (H3).
    """
    failures = [
        event
        for event in events
        if event.status == EventStatus.FAILURE or event.event_type == EventType.ERROR
    ]
    if not failures:
        return None
    failures.sort(key=lambda event: (event.timestamp, str(event.event_id)))
    return str(failures[0].event_id)


def drop_payloads(events) -> tuple[AgentEvent, ...]:
    """
    Return the events with every payload removed (the hostile case).

    This is the most aggressive degradation possible: the payload of *every*
    event is gone, only the structure and the statuses remain.  It also
    destroys the metadata, so retry links cannot be recovered any more.
    """
    return tuple(
        event.model_copy(
            update={
                "input_data": None,
                "output_data": None,
                "metadata": None,
                "observation_level": ObservationLevel.DROP,
            }
        )
        for event in events
    )


#: Event types the observation policy always keeps at FULL (Phase 10, D4).
_ALWAYS_FULL_TYPES = frozenset(
    {EventType.ERROR, EventType.RETRY, EventType.FINAL_RESPONSE}
)


def apply_adaptive_observation(events) -> tuple[AgentEvent, ...]:
    """
    Return the events as adaptive observation mode would have stored them.

    This mirrors the real policy: failed events and the ``ERROR`` /
    ``RETRY`` / ``FINAL_RESPONSE`` types keep their full payload, every
    other event is dropped (payload gone, metadata replaced by the drop
    marker).  The structure of the trace is preserved either way.
    """
    degraded = []
    for event in events:
        keep_full = (
            event.status == EventStatus.FAILURE
            or event.event_type in _ALWAYS_FULL_TYPES
        )
        if keep_full:
            degraded.append(event)
            continue
        degraded.append(
            event.model_copy(
                update={
                    "input_data": None,
                    "output_data": None,
                    "metadata": {"_dropped": True},
                    "observation_level": ObservationLevel.DROP,
                }
            )
        )
    return tuple(degraded)
