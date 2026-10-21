"""Observation policy: decide observation level and transform events.

Design decisions:
- D4: FAILURE status / ERROR / RETRY / FINAL_RESPONSE are always FULL.
- D5: DROP retains skeleton (identity + structure, no payload).
- D6: full mode is a no-op - no transformation, all events FULL.
- D7: Fail-open - errors swallowed by caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentlens.events import AgentEvent, EventStatus, EventType, ObservationLevel


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for the observation policy."""

    mode: str = "full"                  # "full" | "adaptive"
    compress_threshold: float = 0.60    # score >= -> FULL
    metadata_threshold: float = 0.35   # score >= -> COMPRESSED
    drop_threshold: float = 0.15        # score >= -> METADATA_ONLY, else DROP
    payload_max_bytes: int = 4096       # COMPRESSED truncation limit
    always_full_types: frozenset = field(
        default_factory=lambda: frozenset({
            EventType.ERROR,
            EventType.RETRY,
            EventType.FINAL_RESPONSE,
        })
    )
    always_full_statuses: frozenset = field(
        default_factory=lambda: frozenset({EventStatus.FAILURE})
    )
    drop_retain_skeleton: bool = True


class ObservationPolicy:
    """Routes events to an observation level and transforms their payload."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self._cfg = config or PolicyConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, importance_score: float, event: AgentEvent) -> ObservationLevel:
        """Return the ObservationLevel for *event* given its importance score.

        D4 safety overrides run first, regardless of score.
        """
        cfg = self._cfg
        if cfg.mode == "full":
            return ObservationLevel.FULL

        # D4: always-full overrides.
        if event.status in cfg.always_full_statuses:
            return ObservationLevel.FULL
        if event.event_type in cfg.always_full_types:
            return ObservationLevel.FULL

        # Threshold routing.
        if importance_score >= cfg.compress_threshold:
            return ObservationLevel.FULL
        if importance_score >= cfg.metadata_threshold:
            return ObservationLevel.COMPRESSED
        if importance_score >= cfg.drop_threshold:
            return ObservationLevel.METADATA_ONLY
        return ObservationLevel.DROP

    def transform(self, event: AgentEvent, level: ObservationLevel) -> AgentEvent:
        """Return a (possibly mutated copy of) *event* appropriate to *level*.

        - FULL       -> unchanged
        - COMPRESSED -> payload truncated to payload_max_bytes
        - METADATA_ONLY -> input_data and output_data set to None
        - DROP       -> input_data, output_data nulled; metadata={"_dropped": True}
                        (skeleton preserved when drop_retain_skeleton=True)
        """
        if level == ObservationLevel.FULL:
            return event.model_copy(update={"observation_level": ObservationLevel.FULL})

        if level == ObservationLevel.COMPRESSED:
            return event.model_copy(update={
                "observation_level": ObservationLevel.COMPRESSED,
                "input_data": self._truncate(event.input_data),
                "output_data": self._truncate(event.output_data),
            })

        if level == ObservationLevel.METADATA_ONLY:
            return event.model_copy(update={
                "observation_level": ObservationLevel.METADATA_ONLY,
                "input_data": None,
                "output_data": None,
            })

        # DROP
        if self._cfg.drop_retain_skeleton:
            return event.model_copy(update={
                "observation_level": ObservationLevel.DROP,
                "input_data": None,
                "output_data": None,
                "metadata": {"_dropped": True},
            })
        else:
            return event.model_copy(update={
                "observation_level": ObservationLevel.DROP,
                "input_data": None,
                "output_data": None,
                "metadata": None,
            })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate(self, data: dict[str, Any] | None) -> dict[str, Any] | None:
        """Truncate *data* so its JSON representation fits in payload_max_bytes."""
        if data is None:
            return None
        try:
            serialized = json.dumps(data, default=str)
        except Exception:
            return {"_truncated": True, "error": "serialization_failed"}
        max_bytes = self._cfg.payload_max_bytes
        if len(serialized.encode()) <= max_bytes:
            return data
        # Truncate the serialized string and return a summary.
        truncated_str = serialized.encode()[:max_bytes].decode("utf-8", errors="ignore")
        return {
            "_truncated": True,
            "_original_bytes": len(serialized.encode()),
            "_summary": truncated_str[:200],
        }
