"""Deterministic importance scoring for agent events.

Design decisions (from spec):
- D1: Rule-based, no randomness -> reproducible across measurements.
- D2: Pure function - score(event, repeat_index); no side effects.
- D7: Any error is swallowed by the caller (fail-open).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from agentlens.events import AgentEvent, EventStatus, EventType


@dataclass(frozen=True)
class ScoringConfig:
    """Weights and bonuses that drive the importance score."""

    base_by_status: Mapping[EventStatus, float] = field(
        default_factory=lambda: {
            EventStatus.FAILURE: 0.9,
            EventStatus.PARTIAL: 0.7,
            EventStatus.PENDING: 0.6,
            EventStatus.SUCCESS: 0.3,
        }
    )
    weight_by_type: Mapping[EventType, float] = field(
        default_factory=lambda: {
            EventType.ERROR:          1.00,
            EventType.FINAL_RESPONSE: 0.95,
            EventType.RETRY:          0.85,
            EventType.DECISION:       0.80,
            EventType.SUBAGENT_START: 0.70,
            EventType.SUBAGENT_END:   0.70,
            EventType.TOOL_CALL:      0.55,
            EventType.LLM_CALL:       0.50,
            EventType.RETRIEVAL:      0.50,
            EventType.MEMORY_READ:    0.35,
            EventType.MEMORY_WRITE:   0.35,
        }
    )
    parent_bonus: float = 0.10
    latency_high_ms: float = 1000.0
    latency_bonus: float = 0.10
    exception_bonus: float = 0.10
    repetition_penalty: float = 0.15
    repetition_cap: int = 3
    min_score: float = 0.0
    max_score: float = 1.0


class ImportanceScorer:
    """Deterministic importance scorer."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self._cfg = config or ScoringConfig()

    def score(self, event: AgentEvent, *, repeat_index: int = 0) -> float:
        """Return a clamped importance score in [0, 1]."""
        cfg = self._cfg
        base = cfg.base_by_status.get(event.status, 0.5)
        type_weight = cfg.weight_by_type.get(event.event_type, 0.5)
        score = (base + type_weight) / 2.0
        if event.parent_id is None:
            score += cfg.parent_bonus
        if event.latency_ms is not None and event.latency_ms >= cfg.latency_high_ms:
            score += cfg.latency_bonus
        if event.metadata and "exception_type" in event.metadata:
            score += cfg.exception_bonus
        penalty_count = min(repeat_index, cfg.repetition_cap)
        score -= penalty_count * cfg.repetition_penalty
        return max(cfg.min_score, min(cfg.max_score, score))
