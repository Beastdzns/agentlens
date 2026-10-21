"""
Unit tests for the Phase 11 causal graph builder.

Covers the construction rules (D1–D3), the traversal helpers, the
graceful-with-missing-evidence rule (D5) and determinism (D2).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from agentlens.causal.graph import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    CausalRelation,
    build_graph,
)
from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    ObservationLevel,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    *,
    trace_id=None,
    event_type: EventType = EventType.LLM_CALL,
    status: EventStatus = EventStatus.SUCCESS,
    parent_id=None,
    metadata=None,
    input_data=None,
    output_data=None,
    timestamp=None,
    importance_score: float = 0.5,
    observation_level: ObservationLevel = ObservationLevel.FULL,
    agent_id: str = "test-agent",
) -> AgentEvent:
    return AgentEvent(
        trace_id=trace_id or uuid4(),
        agent_id=agent_id,
        event_type=event_type,
        status=status,
        input_data=input_data,
        output_data=output_data,
        parent_id=parent_id,
        metadata=metadata,
        importance_score=importance_score,
        observation_level=observation_level,
        **({"timestamp": timestamp} if timestamp is not None else {}),
    )


def _relations(graph: CausalGraph) -> set[tuple[str, str, str]]:
    return {(e.source, e.target, e.relation.value) for e in graph.edges}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_empty_trace_yields_empty_graph() -> None:
    graph = build_graph([])
    assert graph.nodes == {}
    assert graph.edges == ()
    assert graph.to_dict() == {"nodes": [], "edges": []}


def test_single_event_has_no_edges() -> None:
    event = _event()
    graph = build_graph([event])
    assert len(graph.nodes) == 1
    assert graph.edges == ()
    assert graph.node(str(event.event_id)) is not None


def test_parent_child_creates_parent_of_edge() -> None:
    parent = _event(event_type=EventType.SUBAGENT_START)
    child = _event(event_type=EventType.TOOL_CALL, parent_id=parent.event_id)

    graph = build_graph([child, parent])

    assert _relations(graph) == {(str(parent.event_id), str(child.event_id), "parent_of")}
    assert graph.children(str(parent.event_id)) == [str(child.event_id)]
    assert graph.parents(str(child.event_id)) == [str(parent.event_id)]


def test_unknown_parent_is_ignored_not_invented() -> None:
    """D9: a parent id outside the trace produces no edge, not a phantom node."""
    ghost = uuid4()
    child = _event(parent_id=ghost)

    graph = build_graph([child])

    assert len(graph.nodes) == 1
    assert graph.edges == ()
    assert graph.parents(str(child.event_id)) == []


def test_retry_of_metadata_creates_retry_edge() -> None:
    first = _event(status=EventStatus.FAILURE)
    retry = _event(metadata={"retry_of": str(first.event_id)})

    graph = build_graph([first, retry])

    assert (str(first.event_id), str(retry.event_id), "retry_of") in _relations(graph)


def test_retry_of_unknown_id_is_ignored() -> None:
    retry = _event(metadata={"retry_of": str(uuid4())})
    graph = build_graph([retry])
    assert graph.edges == ()


def test_depends_on_metadata_creates_dependency_edges() -> None:
    dep = _event()
    node = _event(metadata={"depends_on": [str(dep.event_id)]})

    graph = build_graph([dep, node])

    assert (str(node.event_id), str(dep.event_id), "depends_on") in _relations(graph)


def test_depends_on_ignores_non_list_metadata() -> None:
    node = _event(metadata={"depends_on": "not-a-list"})
    graph = build_graph([node])
    assert graph.edges == ()


def test_failed_parent_annotates_children_with_influenced_edge() -> None:
    parent = _event(status=EventStatus.FAILURE)
    child = _event(parent_id=parent.event_id)

    graph = build_graph([parent, child])
    relations = _relations(graph)

    assert (str(parent.event_id), str(child.event_id), "parent_of") in relations
    assert (str(parent.event_id), str(child.event_id), "influenced") in relations


def test_successful_parent_has_no_influenced_edge() -> None:
    parent = _event(status=EventStatus.SUCCESS)
    child = _event(parent_id=parent.event_id)

    graph = build_graph([parent, child])
    relations = _relations(graph)

    assert (str(parent.event_id), str(child.event_id), "parent_of") in relations
    assert not any(rel == "influenced" for _, _, rel in relations)


def test_caused_relation_is_never_constructed() -> None:
    """D3: ``caused`` is the analyzer's verdict, never an input to it."""
    parent = _event(status=EventStatus.FAILURE)
    child = _event(parent_id=parent.event_id)
    graph = build_graph([parent, child])
    assert not any(rel == "caused" for _, _, rel in _relations(graph))


def test_edges_are_deduplicated() -> None:
    parent = _event(status=EventStatus.FAILURE)
    child = _event(parent_id=parent.event_id)
    graph = build_graph([parent, child, child])
    edges = [
        (e.source, e.target, e.relation.value)
        for e in graph.edges
        if e.relation is CausalRelation.PARENT_OF
    ]
    assert edges == [(str(parent.event_id), str(child.event_id), "parent_of")]


# ---------------------------------------------------------------------------
# Node content
# ---------------------------------------------------------------------------


def test_node_carries_observation_fields() -> None:
    event = _event(importance_score=0.83, observation_level=ObservationLevel.COMPRESSED)
    node = build_graph([event]).node(str(event.event_id))

    assert node is not None
    assert node.importance_score == pytest.approx(0.83)
    assert node.observation_level == "compressed"


def test_dropped_event_keeps_structure_without_payload() -> None:
    """D5: a dropped event still exists as a node, without its payload."""
    parent = _event(status=EventStatus.FAILURE)
    child = _event(
        parent_id=parent.event_id,
        observation_level=ObservationLevel.DROP,
        output_data=None,
    )

    graph = build_graph([parent, child])
    node = graph.node(str(child.event_id))

    assert node is not None
    assert node.observation_level == "drop"
    assert node.error_message is None
    assert graph.children(str(parent.event_id)) == [str(child.event_id)]


def test_error_message_is_read_from_recorded_fields() -> None:
    event = _event(status=EventStatus.FAILURE, output_data={"error": "boom"})
    node = build_graph([event]).node(str(event.event_id))
    assert node is not None
    assert node.error_message == "boom"


def test_error_message_falls_back_to_exception_metadata() -> None:
    event = _event(
        status=EventStatus.FAILURE,
        metadata={"exception_message": "ValueError: nope", "exception_type": "ValueError"},
    )
    node = build_graph([event]).node(str(event.event_id))
    assert node is not None
    assert node.error_message == "ValueError: nope"
    assert node.exception_type == "ValueError"


def test_missing_error_message_is_none_not_invented() -> None:
    event = _event(status=EventStatus.FAILURE)
    node = build_graph([event]).node(str(event.event_id))
    assert node is not None
    assert node.error_message is None
    assert node.exception_type is None


def test_successful_step_with_a_problem_in_its_output_is_flagged() -> None:
    """A step can succeed and still hand a wrong result downstream."""
    event = _event(status=EventStatus.SUCCESS, output_data={"error": "no documents"})
    node = build_graph([event]).node(str(event.event_id))
    assert node is not None
    assert node.suspicious_output is True
    assert node.is_failure is False


def test_clean_output_is_not_flagged_as_suspicious() -> None:
    event = _event(status=EventStatus.SUCCESS, output_data={"result": "fine"})
    node = build_graph([event]).node(str(event.event_id))
    assert node is not None
    assert node.suspicious_output is False


def test_label_uses_recorded_tool_name() -> None:
    event = _event(event_type=EventType.TOOL_CALL, output_data={"tool": "search"})
    node = build_graph([event]).node(str(event.event_id))
    assert node is not None
    assert node.label == "tool_call: search"


def test_label_falls_back_to_event_type() -> None:
    event = _event(event_type=EventType.DECISION)
    node = build_graph([event]).node(str(event.event_id))
    assert node is not None
    assert node.label == "decision"


def test_label_is_truncated() -> None:
    event = _event(output_data={"result": "x" * 200})
    node = build_graph([event]).node(str(event.event_id))
    assert node is not None
    assert len(node.label) <= 60
    assert node.label.endswith("…")


def test_failure_flag_marks_failed_nodes() -> None:
    ok = _event(status=EventStatus.SUCCESS)
    bad = _event(status=EventStatus.FAILURE)
    graph = build_graph([ok, bad])
    assert graph.failures() == [str(bad.event_id)]


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def _chain() -> tuple[AgentEvent, AgentEvent, AgentEvent]:
    root = _event(event_type=EventType.SUBAGENT_START)
    mid = _event(parent_id=root.event_id)
    leaf = _event(parent_id=mid.event_id)
    return root, mid, leaf


def test_ancestors_walk_upstream() -> None:
    root, mid, leaf = _chain()
    graph = build_graph([root, mid, leaf])
    assert graph.ancestors(str(leaf.event_id)) == {str(root.event_id), str(mid.event_id)}
    assert graph.ancestors(str(root.event_id)) == set()


def test_descendants_walk_downstream() -> None:
    root, mid, leaf = _chain()
    graph = build_graph([root, mid, leaf])
    assert graph.descendants(str(root.event_id)) == {str(mid.event_id), str(leaf.event_id)}
    assert graph.descendants(str(leaf.event_id)) == set()


def test_traversal_ignores_the_influenced_annotation() -> None:
    """
    The ``influenced`` edge duplicates the ``parent_of`` direction; it must
    never create ancestry that the structure does not have.
    """
    root, mid, leaf = _chain()
    failing = _event(status=EventStatus.FAILURE, parent_id=root.event_id)
    graph = build_graph([root, mid, leaf, failing])

    # The failed node is a child of root, so root is its only ancestor.
    assert graph.ancestors(str(failing.event_id)) == {str(root.event_id)}
    # And the annotation did not invent a descendant for the failed node.
    assert graph.descendants(str(failing.event_id)) == set()


def test_path_is_not_lengthened_by_the_influenced_annotation() -> None:
    root, mid, leaf = _chain()
    graph = build_graph([root, mid, leaf])
    path = graph.path(str(root.event_id), str(leaf.event_id))
    assert path == [str(root.event_id), str(mid.event_id), str(leaf.event_id)]


def test_depth_counts_parent_chain() -> None:
    root, mid, leaf = _chain()
    graph = build_graph([root, mid, leaf])
    assert graph.depth(str(root.event_id)) == 0
    assert graph.depth(str(mid.event_id)) == 1
    assert graph.depth(str(leaf.event_id)) == 2


def test_depth_of_unknown_node_is_zero() -> None:
    graph = build_graph([])
    assert graph.depth(str(uuid4())) == 0


def test_path_returns_shortest_structural_path() -> None:
    root, mid, leaf = _chain()
    graph = build_graph([root, mid, leaf])
    path = graph.path(str(root.event_id), str(leaf.event_id))
    assert path == [str(root.event_id), str(mid.event_id), str(leaf.event_id)]


def test_path_to_unreachable_node_is_none() -> None:
    a = _event()
    b = _event()
    graph = build_graph([a, b])
    assert graph.path(str(a.event_id), str(b.event_id)) is None


def test_path_to_self_is_single_node() -> None:
    a = _event()
    graph = build_graph([a])
    assert graph.path(str(a.event_id), str(a.event_id)) == [str(a.event_id)]


def test_traversal_is_cycle_safe() -> None:
    """A malformed trace with a parent cycle must not hang the traversal."""
    a = _event()
    b = _event(parent_id=a.event_id)
    # Force a cycle: a claims b as its parent.
    cycle_a = a.model_copy(update={"parent_id": b.event_id})
    graph = build_graph([cycle_a, b])

    ancestors = graph.ancestors(str(b.event_id))
    assert str(cycle_a.event_id) in ancestors
    assert len(ancestors) <= 2


# ---------------------------------------------------------------------------
# Determinism & immutability
# ---------------------------------------------------------------------------


def test_graph_is_deterministic_for_shuffled_input() -> None:
    root, mid, leaf = _chain()
    events = [root, mid, leaf]
    forward = build_graph(events).to_dict()
    backward = build_graph(list(reversed(events))).to_dict()
    assert forward == backward


def test_nodes_are_ordered_by_timestamp_then_id() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    late = _event(timestamp=base + timedelta(seconds=10))
    early = _event(timestamp=base)

    graph = build_graph([late, early])
    assert list(graph.nodes) == [str(early.event_id), str(late.event_id)]


def test_build_does_not_mutate_input_events() -> None:
    event = _event(metadata={"retry_of": str(uuid4())})
    before = event.to_dict()
    build_graph([event])
    assert event.to_dict() == before


def test_graph_is_immutable() -> None:
    event = _event()
    graph = build_graph([event])
    with pytest.raises(TypeError):
        graph.nodes["x"] = None  # type: ignore[index]
    with pytest.raises(AttributeError):
        graph.edges = ()  # type: ignore[misc]


def test_to_dict_is_json_serialisable() -> None:
    import json

    root, mid, leaf = _chain()
    payload = build_graph([root, mid, leaf]).to_dict()
    json.dumps(payload)  # must not raise
    assert len(payload["nodes"]) == 3
    assert len(payload["edges"]) == 2


def test_node_and_edge_dicts_have_expected_keys() -> None:
    node = CausalNode(
        event_id="e1",
        parent_id=None,
        event_type="llm_call",
        status="success",
        timestamp="2026-01-01T00:00:00+00:00",
        latency_ms=1.0,
        importance_score=0.5,
        observation_level="full",
        label="llm_call",
    )
    edge = CausalEdge(source="a", target="b", relation=CausalRelation.PARENT_OF)
    assert set(node.to_dict()) == {
        "event_id",
        "parent_id",
        "event_type",
        "status",
        "timestamp",
        "latency_ms",
        "importance_score",
        "observation_level",
        "label",
        "error_message",
        "exception_type",
        "is_failure",
        "suspicious_output",
    }
    assert edge.to_dict() == {"source": "a", "target": "b", "relation": "parent_of"}
