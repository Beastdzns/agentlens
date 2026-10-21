"""
Grounded explanations for a root-cause diagnosis (Phase 11).

The explanation is **templated and deterministic**: it only interpolates
values that were recorded in the trace — event type, status, error message,
exception type and event ids.  No model is called and no cause is inferred
(D1, D7).

Every sentence can be traced back to an event.  When a failed event's payload
was not retained (adaptive observation), the explanation says exactly that
instead of inventing a reason (D5).
"""

from __future__ import annotations

from agentlens.causal.analyzer import (
    CATEGORY_SUSPECTED_UPSTREAM,
    VERDICT_INSUFFICIENT_EVIDENCE,
    VERDICT_NO_FAILURE,
    VERDICT_RECOVERED,
    RootCause,
)
from agentlens.causal.graph import CausalGraph, CausalNode

#: Phrase used when the payload of a failed event was not retained.
NO_DETAIL_PHRASE = "no error detail was retained"

#: Phrase used when an event carries no recorded error message at all.
NO_ERROR_PHRASE = "no error message was recorded"


def _describe(node: CausalNode | None) -> str:
    """A short, recorded-fields-only description of an event."""
    if node is None:
        return "an unknown event"
    label = node.label or node.event_type
    return f"{label} (event_type={node.event_type}, status={node.status})"


def _detail(node: CausalNode | None) -> str:
    """The recorded detail of a failed event, or the reason it is missing."""
    if node is None:
        return NO_DETAIL_PHRASE
    if node.error_message:
        detail = f'error message "{node.error_message}"'
        if node.exception_type:
            detail += f", exception type {node.exception_type}"
        return detail
    if node.exception_type:
        return f"exception type {node.exception_type}"
    if node.observation_level in ("metadata_only", "drop"):
        return NO_DETAIL_PHRASE
    return NO_ERROR_PHRASE


def _path_text(graph: CausalGraph, event_ids: list[str]) -> str:
    """Render an evidence path as ``label -> label -> label``."""
    parts = []
    for event_id in event_ids:
        node = graph.node(event_id)
        parts.append(node.label if node is not None else event_id)
    return " -> ".join(parts)


def explain(root_cause: RootCause, graph: CausalGraph) -> str:
    """
    Return a grounded, human-readable explanation of a diagnosis.

    Args:
        root_cause: The diagnosis to explain.
        graph: The graph the diagnosis was derived from — used to render the
            event labels of the evidence path.

    Returns:
        One to three sentences, containing only recorded values.
    """
    if root_cause.verdict == VERDICT_NO_FAILURE:
        return (
            "No failure was recorded in this trace: every event has a "
            "non-failure status, so there is no root cause to report."
        )

    if root_cause.verdict == VERDICT_RECOVERED:
        recovered = root_cause.recovered_failures
        if not recovered:
            return "The trace recorded a failure that was recovered by a retry."
        first = graph.node(recovered[0])
        return (
            f"The failure in {_describe(first)} was recovered by a later "
            "successful retry, so the trace has no unrecovered root cause. "
            f"The recorded detail: {_detail(first)}."
        )

    node = graph.node(root_cause.root_cause_event_id) if root_cause.root_cause_event_id else None

    if root_cause.verdict == VERDICT_INSUFFICIENT_EVIDENCE:
        if root_cause.category == CATEGORY_SUSPECTED_UPSTREAM:
            return (
                f"No failed ancestor was found for the failing step, but the "
                f"upstream step {_describe(node)} reports a problem in its "
                f"recorded output ({_detail(node)}). It is reported as a "
                "suspected upstream cause, not a confirmed root cause."
            )
        return (
            f"The failing step {_describe(node)} has no failed ancestor and no "
            f"recorded upstream problem. Its recorded detail: {_detail(node)}."
        )

    # verdict == "root_cause"
    affected = len(root_cause.affected_events)
    secondary = len(root_cause.secondary_failures)
    path = _path_text(graph, root_cause.evidence_path)

    sentences = [
        f"The root cause is {_describe(node)}; its recorded detail: "
        f"{_detail(node)}."
    ]
    if len(root_cause.evidence_path) >= 2:
        sentences.append(
            f"The failure propagated along the evidence path {path}, "
            f"affecting {affected} downstream event(s) and {secondary} "
            "secondary failure(s)."
        )
    else:
        sentences.append(
            "No downstream event was recorded as affected, so the failure did "
            "not cascade."
        )
    return " ".join(sentences)
