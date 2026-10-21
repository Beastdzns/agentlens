"""
Rule-based event importance scoring for adaptive observability.

Purpose
-------
Adaptive observability (``report.md`` Phase 10) keeps diagnostic signal while
reducing how much payload is retained. The first half of that pipeline is
**importance scoring**: assign every :class:`~agentlens.events.AgentEvent` a
deterministic importance score in ``[0, 1]``.

Scoring is intentionally:

- **Rule-based** — no ML/RL, so results are reproducible (NFR6).
- **Deterministic** — the same event always scores the same.
- **Pure** — no storage, collector, or network dependencies; the caller (the
  observation engine) supplies the per-trace ``repeat_index``.

Scoring rule
------------
``raw = base_by_status + weight_by_type + modifiers`` then clamped to
``[min_score, max_score]``. Modifiers:

- ``+parent_bonus`` when ``parent_id is None`` (a structural anchor / root step).
- ``+latency_bonus`` when ``latency_ms >= latency_high_ms`` (slow step).
- ``+exception_bonus`` when ``metadata`` carries an ``exception_type``.
- ``-repetition_penalty * min(repeat_index, repetition_cap)`` applied **only**
  to ``SUCCESS`` events that are not ``FINAL_RESPONSE`` or ``DECISION``
  (repetitive benign steps matter less).

Failure / error / retry events naturally score highest: they combine a high
``base_by_status`` with a high ``weight_by_type`` and are never repetition
penalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from agentlens.events import AgentEvent, EventStatus, EventType

#: Event types that are never subject to the repetition penalty even when they
#: succeed repeatedly (they are always diagnostically relevant).
_REPETITION_EXEMPT_TYPES: frozenset[EventType] = frozenset(
    {EventType.FINAL_RESPONSE, EventType.DECISION}
)


def _default_base_by_status() -> dict[EventStatus, float]:
    """Baseline contribution by event status (better status → lower score)."""
    return {
        EventStatus.FAILURE: 0.90,
        EventStatus.PARTIAL: 0.70,
        EventStatus.PENDING: 0.60,
        EventStatus.SUCCESS: 0.30,
    }


def _default_weight_by_type() -> dict[EventType, float]:
    """Contribution by event type (more diagnostic → higher weight)."""
    return {
        EventType.ERROR: 1.00,
        EventType.FINAL_RESPONSE: 0.95,
        EventType.RETRY: 0.85,
        EventType.DECISION: 0.80,
        EventType.SUBAGENT_START: 0.70,
        EventType.SUBAGENT_END: 0.70,
        EventType.TOOL_CALL: 0.55,
        EventType.LLM_CALL: 0.50,
        EventType.RETRIEVAL: 0.50,
        EventType.MEMORY_WRITE: 0.35,
        EventType.MEMORY_READ: 0.35,
    }


@dataclass(frozen=True)
class ScoringConfig:
    """
    Tunable parameters for :class:`ImportanceScorer`.

    All contributions are additive; the final score is clamped to
    ``[min_score, max_score]``. Override individual fields to customise the
    scoring logic without subclassing.
    """

    base_by_status: Mapping[EventStatus, float] = field(
        default_factory=_default_base_by_status
    )
    weight_by_type: Mapping[EventType, float] = field(
        default_factory=_default_weight_by_type
    )
    parent_bonus: float = 0.10
    latency_high_ms: float = 1000.0
    latency_bonus: float = 0.10
    exception_bonus: float = 0.10
    repetition_penalty: float = 0.15
    repetition_cap: int = 3
    min_score: float = 0.0
    max_score: float = 1.0


#: Shared default configuration used when the caller does not supply one.
DEFAULT_SCORING_CONFIG = ScoringConfig()


class ImportanceScorer:
    """
    Assign deterministic importance scores to :class:`AgentEvent` instances.

    Example::

        scorer = ImportanceScorer()
        score = scorer.score(event)                    # first occurrence
        score = scorer.score(event, repeat_index=2)    # 3rd identical-type event
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self._config = config or DEFAULT_SCORING_CONFIG

    @property
    def config(self) -> ScoringConfig:
        """Return the configuration used by this scorer."""
        return self._config

    def score(self, event: AgentEvent, *, repeat_index: int = 0) -> float:
        """
        Compute the importance score for ``event``.

        Args:
            event: The event to score. Never mutated.
            repeat_index: How many events of the same ``event_type`` have
                already been recorded in this trace (``0`` for the first).
                Used for the repetition penalty; negative values are treated
                as ``0``.

        Returns:
            The importance score, clamped to the configured ``[min, max]``
            range (``[0, 1]`` by default).
        """
        cfg = self._config
        score = cfg.base_by_status.get(event.status, 0.0)
        score += cfg.weight_by_type.get(event.event_type, 0.0)

        if event.parent_id is None:
            score += cfg.parent_bonus

        if event.latency_ms is not None and event.latency_ms >= cfg.latency_high_ms:
            score += cfg.latency_bonus

        if event.metadata and event.metadata.get("exception_type"):
            score += cfg.exception_bonus

        if self._is_repetition_penalised(event) and repeat_index > 0:
            score -= cfg.repetition_penalty * min(repeat_index, cfg.repetition_cap)

        return max(cfg.min_score, min(cfg.max_score, score))

    @staticmethod
    def _is_repetition_penalised(event: AgentEvent) -> bool:
        """Return True when a repeated event of this shape loses importance."""
        return (
            event.status == EventStatus.SUCCESS
            and event.event_type not in _REPETITION_EXEMPT_TYPES
        )
