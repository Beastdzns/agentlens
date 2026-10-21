"""
Root-cause analysis over a causal graph (Phase 11).

Purpose
-------
The causal graph says *how* the steps of a trace are related.  This module
says **which step caused the failure** — and it does so with rules, not with
a model: the same graph always yields the same diagnosis (D1).

How the verdict is reached
--------------------------
1. **Failures** are the nodes whose status is ``failure`` (an ``error``-typed
   node counts as a failure too).
2. A failure with **no failed ancestor** is *primary* — nothing else in the
   trace failed before it.  A failure with a failed ancestor is *secondary*:
   it is a consequence.
3. The **root cause** is the primary failure that affected the most events.
   Ties are broken deterministically: deeper first, then earlier timestamp,
   then the lexicographically smaller event id.
4. A failure that was followed by a **successful retry** is *recovered* — it
   is reported, but it is not an unrecovered root cause.
5. When no failure exists at all the verdict is ``no_failure``; when every
   failure was recovered it is ``recovered``.  The analyzer never guesses
   (D6).

Grounding
---------
Every statement the analyzer makes is traceable to a recorded field: event
type, status, error message, exception type, event ids.  Nothing is inferred
and no model is called (D1, D7).  When the payload of a failed event was not
retained (adaptive observation), the analyzer says so and lowers its
confidence instead of inventing a cause (D5).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from agentlens.causal.graph import CausalGraph, CausalNode, CausalRelation

#: Verdict values.
VERDICT_ROOT_CAUSE = "root_cause"
VERDICT_NO_FAILURE = "no_failure"
VERDICT_RECOVERED = "recovered"
VERDICT_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

#: Category values.
CATEGORY_TOOL_ERROR = "tool_error"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_INVALID_INPUT = "invalid_input"
CATEGORY_LLM_ERROR = "llm_error"
CATEGORY_SUSPECTED_UPSTREAM = "suspected_upstream"
CATEGORY_UNKNOWN = "unknown"
CATEGORY_NONE = "none"

#: Relations that express structure (mirrors the graph's structural set).
_STRUCTURAL_RELATIONS = frozenset(
    {
        CausalRelation.PARENT_OF.value,
        CausalRelation.RETRY_OF.value,
        CausalRelation.DEPENDS_ON.value,
    }
)


@dataclass(frozen=True)
class AnalyzerConfig:
    """
    Tunables of the analyzer.

    The defaults encode the frozen rules; they are exposed so a caller can
    tighten or loosen them without forking the algorithm.
    """

    #: Statuses that mark a node as failed.
    failure_statuses: frozenset[str] = frozenset({"failure"})
    #: Event types that mark a node as failed regardless of its status.
    failure_event_types: frozenset[str] = frozenset({"error"})
    #: Error-message markers that identify a timeout.
    timeout_markers: tuple[str, ...] = ("timeout", "timed out")
    #: Error-message markers that identify invalid input.
    invalid_input_markers: tuple[str, ...] = ("invalid", "bad request", "400")
    #: Observation levels whose events no longer carry their payload.
    degraded_levels: frozenset[str] = frozenset({"metadata_only", "drop"})
    #: Confidence ceiling for a root cause whose payload was not retained.
    degraded_confidence_cap: float = 0.5


@dataclass(frozen=True)
class RootCause:
    """The diagnosis of a trace."""

    verdict: str
    root_cause_event_id: Optional[str]
    confidence: float
    category: str
    explanation: str
    evidence_path: list[str] = field(default_factory=list)
    primary_failures: list[str] = field(default_factory=list)
    secondary_failures: list[str] = field(default_factory=list)
    affected_events: list[str] = field(default_factory=list)
    recovered_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "verdict": self.verdict,
            "root_cause_event_id": self.root_cause_event_id,
            "confidence": self.confidence,
            "category": self.category,
            "explanation": self.explanation,
            "evidence_path": list(self.evidence_path),
            "primary_failures": list(self.primary_failures),
            "secondary_failures": list(self.secondary_failures),
            "affected_events": list(self.affected_events),
            "recovered_failures": list(self.recovered_failures),
        }


# ---------------------------------------------------------------------------
# Graph helpers (kept inside S's ownership boundary)
# ---------------------------------------------------------------------------


def _structural_edges(graph: CausalGraph) -> list[Any]:
    """The edges that express structure."""
    return [e for e in graph.edges if e.relation.value in _STRUCTURAL_RELATIONS]


def _upstream_distances(graph: CausalGraph, event_id: str) -> dict[str, int]:
    """
    Distance from ``event_id`` to every ancestor (1 = direct parent).

    Breadth-first, so the closest ancestor always has the smallest distance.
    """
    reverse: dict[str, list[str]] = {}
    for edge in _structural_edges(graph):
        reverse.setdefault(edge.target, []).append(edge.source)

    distances: dict[str, int] = {}
    queue: list[str] = [event_id]
    visited = {event_id}
    depth = 0
    while queue:
        depth += 1
        next_queue: list[str] = []
        for current in queue:
            for ancestor in sorted(reverse.get(current, ())):
                if ancestor in visited:
                    continue
                visited.add(ancestor)
                distances[ancestor] = depth
                next_queue.append(ancestor)
        queue = next_queue
    return distances


def _retry_successors(graph: CausalGraph) -> dict[str, list[str]]:
    """Map each event to the events that retried it."""
    successors: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.relation is CausalRelation.RETRY_OF:
            successors.setdefault(edge.source, []).append(edge.target)
    return successors


def _is_failure(node: CausalNode, config: AnalyzerConfig) -> bool:
    """A node failed when its status says so, or when it is an error event."""
    if node.status.lower() in config.failure_statuses:
        return True
    return node.event_type.lower() in config.failure_event_types


def _is_degraded(node: CausalNode, config: AnalyzerConfig) -> bool:
    """True when the event's payload was not retained (D5)."""
    return node.observation_level.lower() in config.degraded_levels


def _looks_wrong(node: Optional[CausalNode]) -> bool:
    """
    True when a node's recorded output shows that something went wrong.

    Only recorded evidence is inspected: a reported problem in its output, a
    recorded error message or a recorded exception type.
    """
    if node is None:
        return False
    return bool(node.suspicious_output or node.error_message or node.exception_type)


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------


class RootCauseAnalyzer:
    """
    Rule-based root-cause analyzer.

    Usage::

        graph = build_graph(events)
        diagnosis = RootCauseAnalyzer().analyze(graph)
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.config = config or AnalyzerConfig()

    # -- public API ---------------------------------------------------------

    def analyze(self, graph: CausalGraph) -> RootCause:
        """Diagnose the failures of ``graph``."""
        config = self.config

        failures = [node for node in graph.nodes.values() if _is_failure(node, config)]
        failure_ids = {node.event_id for node in failures}

        if not failures:
            return self._finish(
                graph,
                RootCause(
                    verdict=VERDICT_NO_FAILURE,
                    root_cause_event_id=None,
                    confidence=0.0,
                    category=CATEGORY_NONE,
                    explanation="",
                ),
            )

        recovered = self._recovered_failures(graph, failure_ids)
        unrecovered = [n for n in failures if n.event_id not in recovered]
        primary, secondary = self._split_failures(graph, failure_ids)

        if not unrecovered:
            # Every failure was followed by a successful retry (D6).
            return self._finish(
                graph,
                RootCause(
                    verdict=VERDICT_RECOVERED,
                    root_cause_event_id=None,
                    confidence=0.0,
                    category=CATEGORY_NONE,
                    explanation="",
                    primary_failures=[],
                    secondary_failures=sorted(secondary),
                    affected_events=[],
                    recovered_failures=sorted(recovered),
                ),
            )

        candidates = [n for n in primary if n.event_id not in recovered] or primary
        root = self._select_root(graph, candidates)
        terminal = self._terminal_failure(graph, failures)

        # -- the causal chain ------------------------------------------------
        path = graph.path(root.event_id, terminal.event_id)
        evidence_path = list(path) if path else [root.event_id]

        suspicious: Optional[CausalNode] = None
        if not (graph.ancestors(terminal.event_id) & failure_ids):
            # No failed ancestor: look for an upstream step whose recorded
            # output looks wrong (rule 9).
            suspicious = self._suspect_upstream(graph, terminal.event_id)
            if suspicious is not None:
                suspicious_path = graph.path(suspicious.event_id, terminal.event_id)
                evidence_path = (
                    list(suspicious_path)
                    if suspicious_path
                    else [suspicious.event_id]
                )

        # -- verdict, category, confidence -----------------------------------
        if suspicious is not None:
            verdict = VERDICT_INSUFFICIENT_EVIDENCE
            reported = suspicious
            category = CATEGORY_SUSPECTED_UPSTREAM
            confidence = 0.5
        else:
            verdict = VERDICT_ROOT_CAUSE
            reported = root
            category = self._category(root)
            if len(evidence_path) >= 2:
                confidence = 1.0 if root.exception_type else 0.8
            else:
                confidence = 0.3

        # -- D5: a degraded payload caps the confidence ----------------------
        if _is_degraded(reported, config):
            confidence = min(confidence, config.degraded_confidence_cap)

        affected = sorted(
            node_id
            for node_id in graph.descendants(reported.event_id)
            if node_id not in failure_ids
        )

        return self._finish(
            graph,
            RootCause(
                verdict=verdict,
                root_cause_event_id=reported.event_id,
                confidence=round(confidence, 3),
                category=category,
                explanation="",
                evidence_path=evidence_path,
                primary_failures=sorted(node.event_id for node in primary),
                secondary_failures=sorted(secondary),
                affected_events=affected,
                recovered_failures=sorted(recovered),
            ),
        )

    # -- internals ----------------------------------------------------------

    def _finish(self, graph: CausalGraph, result: RootCause) -> RootCause:
        """Fill in the grounded explanation."""
        from agentlens.causal.explain import explain

        return replace(result, explanation=explain(result, graph))

    def _recovered_failures(self, graph: CausalGraph, failure_ids: set[str]) -> set[str]:
        """Failures that were followed by a successful retry."""
        successors = _retry_successors(graph)
        recovered: set[str] = set()
        for failure_id in sorted(failure_ids):
            queue = list(successors.get(failure_id, ()))
            visited: set[str] = set()
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current not in failure_ids:
                    recovered.add(failure_id)
                    break
                queue.extend(successors.get(current, ()))
        return recovered

    def _split_failures(
        self, graph: CausalGraph, failure_ids: set[str]
    ) -> tuple[list[CausalNode], list[str]]:
        """Split failures into primary (no failed ancestor) and secondary."""
        primary: list[CausalNode] = []
        secondary: list[str] = []
        for failure_id in sorted(failure_ids):
            node = graph.node(failure_id)
            if node is None:
                continue
            if graph.ancestors(failure_id) & failure_ids:
                secondary.append(failure_id)
            else:
                primary.append(node)
        return primary, secondary

    def _select_root(
        self, graph: CausalGraph, candidates: list[CausalNode]
    ) -> CausalNode:
        """Pick the candidate that affected the most events (deterministic)."""

        def sort_key(node: CausalNode) -> tuple[int, int, str, str]:
            descendants = len(graph.descendants(node.event_id))
            # Most descendants first, then deepest, then earliest, then id.
            return (
                -descendants,
                -graph.depth(node.event_id),
                node.timestamp,
                node.event_id,
            )

        return sorted(candidates, key=sort_key)[0]

    def _terminal_failure(self, graph: CausalGraph, failures: list[CausalNode]) -> CausalNode:
        """The deepest failure of the tree (tie-break: earlier, then id)."""
        return sorted(
            failures,
            key=lambda node: (-graph.depth(node.event_id), node.timestamp, node.event_id),
        )[0]

    def _suspect_upstream(
        self, graph: CausalGraph, terminal_id: str
    ) -> Optional[CausalNode]:
        """Find the closest ancestor whose recorded output looks wrong."""
        distances = _upstream_distances(graph, terminal_id)
        for ancestor_id, _distance in sorted(distances.items(), key=lambda kv: kv[1]):
            node = graph.node(ancestor_id)
            if _looks_wrong(node):
                return node
        return None

    def _category(self, node: CausalNode) -> str:
        """Derive the category from the node's recorded fields only."""
        config = self.config
        message = (node.error_message or "").lower()
        if any(marker in message for marker in config.timeout_markers):
            return CATEGORY_TIMEOUT
        if any(marker in message for marker in config.invalid_input_markers):
            return CATEGORY_INVALID_INPUT
        if node.event_type.lower() == "tool_call":
            return CATEGORY_TOOL_ERROR
        if node.event_type.lower() == "llm_call":
            return CATEGORY_LLM_ERROR
        return CATEGORY_UNKNOWN
