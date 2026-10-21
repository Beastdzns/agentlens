"""
Causal graph construction for agent traces (Phase 11).

Purpose
-------
When an agent fails, the useful question is not *"which event failed?"* but
*"what caused it?"*.  Answering that needs an explicit graph of the trace:
which step contained which other step, which attempt was a retry of which,
and which failure propagated to which descendants.

This module turns a flat list of :class:`~agentlens.events.AgentEvent` into
that graph.  It is deliberately **pure** (D2): no I/O, no mutation of the
events it is given, no dependency on storage, the collector or the API.  The
same events always produce the same graph.

Design
------
**Closed edge taxonomy (D3).**  Only four relation types exist:

=============  ==========================  ==================================
Relation       Direction                   Derived from
=============  ==========================  ==================================
``parent_of``   parent → child              ``event.parent_id``
``retry_of``    failed attempt → retry     ``metadata["retry_of"]``
``influenced``  failed node → child         a failing parent (annotation)
``depends_on``  node → dependency           ``metadata["depends_on"]``
=============  ==========================  ==================================

``caused`` is deliberately **not** a construction edge — it is the root-cause
analyzer's verdict, and never an input to it (D3).

**Traversal is structural.**  ``ancestors()`` / ``descendants()`` follow the
structural relations (``parent_of``, ``retry_of``, ``depends_on``).  The
``influenced`` edge duplicates the ``parent_of`` direction and exists purely so
the UI can draw failure propagation differently.

**Unresolvable references are ignored, never invented** (D9).  A parent id
that is not part of the trace, or a ``retry_of`` id that was never recorded,
simply produces no edge.

**Graceful with missing evidence (D5).**  In adaptive observation mode an
event may have been dropped, so its payload is gone.  The graph still carries
the node — with ``observation_level`` and an empty ``error_message`` — so the
causal structure survives even when the payload does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Optional

#: Statuses that count as a failure for graph purposes.
_FAILURE_STATUSES = frozenset({"failure", "error"})


class CausalRelation(str, Enum):
    """The closed set of causal relations (D3)."""

    PARENT_OF = "parent_of"
    RETRY_OF = "retry_of"
    INFLUENCED = "influenced"
    DEPENDS_ON = "depends_on"


#: Relations that express structure and are therefore traversed.
_STRUCTURAL_RELATIONS = frozenset(
    {
        CausalRelation.PARENT_OF.value,
        CausalRelation.RETRY_OF.value,
        CausalRelation.DEPENDS_ON.value,
    }
)

#: Maximum number of characters kept from a payload value in a node label.
_LABEL_SUMMARY_CHARS = 40

#: ``output_data`` keys whose presence signals that a step produced a wrong
#: or failed result even though the step itself did not fail.
_SUSPICIOUS_OUTPUT_KEYS = ("error", "exception", "invalid")


@dataclass(frozen=True)
class CausalNode:
    """A single event rendered as a node of the causal graph."""

    event_id: str
    parent_id: Optional[str]
    event_type: str
    status: str
    timestamp: str
    latency_ms: Optional[float]
    importance_score: float
    observation_level: str
    label: str
    error_message: Optional[str] = None
    exception_type: Optional[str] = None
    #: True when the recorded output of the step reports a problem (the step
    #: itself may still have succeeded) — used by the root-cause analyzer to
    #: spot a suspicious upstream step.
    suspicious_output: bool = False

    @property
    def is_failure(self) -> bool:
        """True when this node represents a failed step."""
        return self.status.lower() in _FAILURE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "event_id": self.event_id,
            "parent_id": self.parent_id,
            "event_type": self.event_type,
            "status": self.status,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "importance_score": self.importance_score,
            "observation_level": self.observation_level,
            "label": self.label,
            "error_message": self.error_message,
            "exception_type": self.exception_type,
            "is_failure": self.is_failure,
            "suspicious_output": self.suspicious_output,
        }


@dataclass(frozen=True)
class CausalEdge:
    """A directed relation between two nodes."""

    source: str
    target: str
    relation: CausalRelation

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
        }


@dataclass(frozen=True)
class CausalGraph:
    """
    An immutable causal graph over the events of one trace.

    The traversal helpers are the interface the root-cause analyzer builds on
    (see :mod:`agentlens.causal.analyzer`).
    """

    nodes: dict[str, CausalNode]
    edges: tuple[CausalEdge, ...]

    # Derived adjacency maps (rebuilt from ``edges`` on construction).
    _adjacency: dict[str, list[str]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _reverse_adjacency: dict[str, list[str]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _children: dict[str, list[str]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _parents: dict[str, list[str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Derive the adjacency maps used by the traversal helpers."""
        adjacency: dict[str, list[str]] = {}
        reverse: dict[str, list[str]] = {}
        children: dict[str, list[str]] = {}
        parents: dict[str, list[str]] = {}

        for edge in self.edges:
            if edge.relation.value in _STRUCTURAL_RELATIONS:
                adjacency.setdefault(edge.source, []).append(edge.target)
                reverse.setdefault(edge.target, []).append(edge.source)
            if edge.relation is CausalRelation.PARENT_OF:
                children.setdefault(edge.source, []).append(edge.target)
                parents.setdefault(edge.target, []).append(edge.source)

        object.__setattr__(
            self, "_adjacency", {k: sorted(v) for k, v in adjacency.items()}
        )
        object.__setattr__(
            self, "_reverse_adjacency", {k: sorted(v) for k, v in reverse.items()}
        )
        object.__setattr__(self, "_children", {k: sorted(v) for k, v in children.items()})
        object.__setattr__(self, "_parents", {k: sorted(v) for k, v in parents.items()})
        # The node map is exposed read-only: a graph is a snapshot, not a
        # mutable view onto the events it was built from.
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))

    # -- basic accessors ----------------------------------------------------

    def node(self, event_id: str) -> Optional[CausalNode]:
        """Return the node with ``event_id``, or ``None``."""
        return self.nodes.get(event_id)

    def children(self, event_id: str) -> list[str]:
        """Return the ids of the direct structural children of a node."""
        return list(self._children.get(event_id, ()))

    def parents(self, event_id: str) -> list[str]:
        """Return the ids of the direct structural parents of a node."""
        return list(self._parents.get(event_id, ()))

    def failures(self) -> list[str]:
        """Return the ids of every failed node, in graph order."""
        return [nid for nid, node in self.nodes.items() if node.is_failure]

    # -- traversal ----------------------------------------------------------

    def ancestors(self, event_id: str) -> set[str]:
        """Return every node reachable *upstream* of ``event_id``."""
        return self._reachable(event_id, reverse=True)

    def descendants(self, event_id: str) -> set[str]:
        """Return every node reachable *downstream* of ``event_id``."""
        return self._reachable(event_id, reverse=False)

    def _reachable(self, event_id: str, *, reverse: bool) -> set[str]:
        """Breadth-first traversal over structural edges (cycle-safe)."""
        adjacency = self._reverse_adjacency if reverse else self._adjacency
        seen: set[str] = set()
        queue = list(adjacency.get(event_id, ()))
        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(adjacency.get(current, ()))
        return seen

    def depth(self, event_id: str) -> int:
        """
        Return the ancestor depth of a node (roots are ``0``).

        Depth is the length of the longest parent chain to a root, so a node
        that hangs off a deep chain scores higher than a sibling that does
        not.  Cycle-safe and iterative.
        """
        if event_id not in self.nodes:
            return 0

        memo: dict[str, int] = {}
        for start in self.nodes:
            if start in memo:
                continue
            # Walk up from ``start`` until a memoised node or a root is hit.
            stack: list[str] = []
            current = start
            while current not in memo:
                if current not in self.nodes:
                    break
                stack.append(current)
                parents = self.parents(current)
                if not parents:
                    break
                current = parents[0]
            # ``memo[current]`` is the depth of the nearest known ancestor
            # (0 for a root); each node on the stack is one level deeper.
            base = memo.get(current, 0)
            for nid in reversed(stack):
                base = base + 1 if self.parents(nid) else 0
                memo[nid] = base
        return memo.get(event_id, 0)

    def path(self, source_id: str, target_id: str) -> Optional[list[str]]:
        """
        Return the shortest structural path ``source_id → target_id``.

        Returns ``None`` when the target is not reachable from the source.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        if source_id == target_id:
            return [source_id]

        previous: dict[str, Optional[str]] = {source_id: None}
        queue = [source_id]
        while queue:
            current = queue.pop(0)
            for neighbour in self._adjacency.get(current, ()):
                if neighbour in previous:
                    continue
                previous[neighbour] = current
                if neighbour == target_id:
                    path: list[str] = []
                    cursor: Optional[str] = target_id
                    while cursor is not None:
                        path.append(cursor)
                        cursor = previous[cursor]
                    path.reverse()
                    return path
                queue.append(neighbour)
        return None

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the whole graph."""
        return {
            "nodes": [self.nodes[nid].to_dict() for nid in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _status_of(event: Any) -> str:
    status = getattr(event, "status", "success")
    return str(getattr(status, "value", status))


def _event_type_of(event: Any) -> str:
    event_type = getattr(event, "event_type", "unknown")
    return str(getattr(event_type, "value", event_type))


def _timestamp_of(event: Any) -> str:
    timestamp = getattr(event, "timestamp", None)
    if timestamp is None:
        return ""
    return str(timestamp)


def _metadata_of(event: Any) -> dict[str, Any]:
    metadata = getattr(event, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _payload_of(event: Any, name: str) -> dict[str, Any]:
    payload = getattr(event, name, None)
    return payload if isinstance(payload, dict) else {}


def _error_message_of(event: Any) -> Optional[str]:
    """Extract the recorded error message, if any (never invented)."""
    for source, key in (
        (_payload_of(event, "output_data"), "error"),
        (_metadata_of(event), "exception_message"),
        (_payload_of(event, "output_data"), "message"),
    ):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _exception_type_of(event: Any) -> Optional[str]:
    value = _metadata_of(event).get("exception_type")
    return value if isinstance(value, str) and value.strip() else None


def _suspicious_output_of(event: Any) -> bool:
    """
    True when the recorded output of the step reports a problem.

    A step can succeed and still hand a wrong result downstream — the
    recorded output keys are the only evidence of that.
    """
    output = _payload_of(event, "output_data")
    if any(key in output for key in _SUSPICIOUS_OUTPUT_KEYS):
        return True
    return bool(_metadata_of(event).get("exception_type"))


def _short(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) > _LABEL_SUMMARY_CHARS:
        return text[:_LABEL_SUMMARY_CHARS] + "…"
    return text


def _label_for(event: Any, event_type: str) -> str:
    """Build a short, recorded-fields-only label for a node."""
    metadata = _metadata_of(event)
    output = _payload_of(event, "output_data")
    input_data = _payload_of(event, "input_data")

    summary: Optional[str] = None
    for candidate in (
        output.get("tool"),
        output.get("result"),
        output.get("answer"),
        input_data.get("query"),
        input_data.get("question"),
    ):
        summary = _short(candidate)
        if summary:
            break

    attempt = metadata.get("attempt")
    if isinstance(attempt, int) and attempt > 1:
        summary = f"attempt {attempt}" + (f" — {summary}" if summary else "")

    return f"{event_type}: {summary}" if summary else event_type


def _node_for(event: Any) -> CausalNode:
    event_type = _event_type_of(event)
    event_id = str(getattr(event, "event_id", ""))
    parent_id = getattr(event, "parent_id", None)
    return CausalNode(
        event_id=event_id,
        parent_id=str(parent_id) if parent_id else None,
        event_type=event_type,
        status=_status_of(event),
        timestamp=_timestamp_of(event),
        latency_ms=getattr(event, "latency_ms", None),
        importance_score=float(getattr(event, "importance_score", 0.5) or 0.0),
        observation_level=str(
            getattr(getattr(event, "observation_level", "full"), "value", "full")
        ),
        label=_label_for(event, event_type),
        error_message=_error_message_of(event),
        exception_type=_exception_type_of(event),
        suspicious_output=_suspicious_output_of(event),
    )


def _parent_edges(nodes: dict[str, CausalNode]) -> set[tuple[str, str, CausalRelation]]:
    """Structural containment edges (``parent_of``)."""
    return {
        (node.parent_id, node.event_id, CausalRelation.PARENT_OF)
        for node in nodes.values()
        if node.parent_id and node.parent_id in nodes
    }


def _retry_edges(
    events: list[Any], nodes: dict[str, CausalNode]
) -> set[tuple[str, str, CausalRelation]]:
    """Retry edges (``retry_of``) — the attempt a retry followed up on."""
    edges: set[tuple[str, str, CausalRelation]] = set()
    for event in events:
        event_id = str(getattr(event, "event_id", ""))
        retry_of = _metadata_of(event).get("retry_of")
        if event_id and isinstance(retry_of, str) and retry_of in nodes:
            edges.add((retry_of, event_id, CausalRelation.RETRY_OF))
    return edges


def _dependency_edges(
    events: list[Any], nodes: dict[str, CausalNode]
) -> set[tuple[str, str, CausalRelation]]:
    """Explicit dependency edges (``depends_on``) recorded by the agent."""
    edges: set[tuple[str, str, CausalRelation]] = set()
    for event in events:
        event_id = str(getattr(event, "event_id", ""))
        depends_on = _metadata_of(event).get("depends_on")
        if not event_id or not isinstance(depends_on, (list, tuple)):
            continue
        for dependency in depends_on:
            if isinstance(dependency, str) and dependency in nodes:
                edges.add((event_id, dependency, CausalRelation.DEPENDS_ON))
    return edges


def _influenced_edges(
    nodes: dict[str, CausalNode],
    parent_edges: set[tuple[str, str, CausalRelation]],
) -> set[tuple[str, str, CausalRelation]]:
    """Annotation edges (``influenced``) from a failing step to its children."""
    children: dict[str, list[str]] = {}
    for src, dst, rel in parent_edges:
        if rel is CausalRelation.PARENT_OF:
            children.setdefault(src, []).append(dst)
    return {
        (node.event_id, child_id, CausalRelation.INFLUENCED)
        for node in nodes.values()
        if node.is_failure
        for child_id in children.get(node.event_id, ())
    }


def build_graph(events: Iterable[Any]) -> CausalGraph:
    """
    Build the causal graph of a trace.

    Args:
        events: The events of a single trace, in any order.  They are never
            mutated.

    Returns:
        A :class:`CausalGraph` whose nodes are ordered by ``(timestamp,
        event_id)`` and whose edges are ordered by ``(relation, source,
        target)`` — so the result is deterministic (D2).
    """
    event_list = list(events)

    nodes: dict[str, CausalNode] = {}
    for event in event_list:
        node = _node_for(event)
        if node.event_id:
            nodes[node.event_id] = node

    parent_edges = _parent_edges(nodes)
    edges = (
        parent_edges
        | _retry_edges(event_list, nodes)
        | _dependency_edges(event_list, nodes)
        | _influenced_edges(nodes, parent_edges)
    )

    ordered_edges = tuple(
        CausalEdge(source=src, target=dst, relation=rel)
        for src, dst, rel in sorted(edges, key=lambda e: (e[2].value, e[0], e[1]))
    )

    ordered_nodes = {
        nid: nodes[nid] for nid in sorted(nodes, key=lambda n: (nodes[n].timestamp, n))
    }

    return CausalGraph(nodes=ordered_nodes, edges=ordered_edges)
