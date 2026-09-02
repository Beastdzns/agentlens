"""
Event normalization adapter for AgentLens.

:class:`EventNormalizer` converts arbitrary dictionaries (and, in future,
framework-specific event objects) into validated :class:`~agentlens.events.AgentEvent`
instances.

Responsibilities
----------------
- **Field aliasing** — maps common alternative names to canonical fields
  (e.g. ``response`` → ``output_data``, ``prompt`` → ``input_data``).
- **Timestamp normalization** — accepts ISO 8601 strings, Unix epoch floats,
  and naïve/aware :class:`~datetime.datetime` objects; all are coerced to
  timezone-aware UTC.
- **Safe defaults** — fills required fields (``trace_id``, ``agent_id``,
  ``event_type``) with sensible fallbacks so the normalizer is fail-open when
  called with incomplete data.
- **Framework stubs** — ``from_langchain_event`` and ``from_llamaindex_event``
  are forward stubs ready for Vyankatesh or the next sprint.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from agentlens.events import AgentEvent, EventStatus, EventType, ObservationLevel

# ---------------------------------------------------------------------------
# Field alias map
# ---------------------------------------------------------------------------

#: Maps alternative/framework-specific keys → canonical AgentEvent field names.
_FIELD_ALIASES: dict[str, str] = {
    # Input data aliases
    "prompt": "input_data",
    "query": "input_data",
    "inputs": "input_data",
    "request": "input_data",
    # Output data aliases
    "response": "output_data",
    "output": "output_data",
    "result": "output_data",
    "outputs": "output_data",
    "answer": "output_data",
    # Event type aliases
    "type": "event_type",
    "kind": "event_type",
    "operation": "event_type",
    # Status aliases
    "state": "status",
    # Latency aliases
    "duration_ms": "latency_ms",
    "elapsed_ms": "latency_ms",
    "latency": "latency_ms",
    # Agent aliases
    "agent": "agent_id",
    "source": "agent_id",
}

# ---------------------------------------------------------------------------
# Timestamp normalization helpers
# ---------------------------------------------------------------------------


def _normalize_timestamp(value: Any) -> datetime:
    """
    Convert *value* to a timezone-aware UTC :class:`datetime`.

    Supported formats:

    * ``datetime`` (naïve → assumed UTC, aware → converted to UTC)
    * ``float`` / ``int`` — interpreted as Unix epoch seconds
    * ``str`` — parsed via :meth:`datetime.fromisoformat`; if the resulting
      object is naïve, UTC is assumed.

    Falls back to ``datetime.now(timezone.utc)`` on any parse error.

    Args:
        value: Raw timestamp value in any supported format.

    Returns:
        A timezone-aware UTC :class:`datetime`.
    """
    try:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)

        if isinstance(value, str):
            # Handle both "Z" suffix (Python < 3.11 doesn't support it) and "+00:00"
            clean = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(clean)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"AgentLens EventNormalizer: could not parse timestamp {value!r}: {exc!r}. "
            "Falling back to datetime.now(UTC).",
            RuntimeWarning,
            stacklevel=3,
        )

    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# EventNormalizer
# ---------------------------------------------------------------------------


class EventNormalizer:
    """
    Adapter that normalizes arbitrary event dictionaries into
    :class:`~agentlens.events.AgentEvent` instances.

    All conversion methods are **fail-open**: missing required fields receive
    safe defaults (auto-generated UUIDs, ``"unknown"`` agent ID, ``"llm_call"``
    event type) so that a malformed upstream event never crashes the agent.

    Example::

        normalizer = EventNormalizer()

        event = normalizer.from_dict({
            "agent_id": "research-bot",
            "event_type": "tool_call",
            "prompt": {"query": "ai agents"},
            "response": {"hits": 42},
            "latency_ms": 88.0,
            "trace_id": "550e8400-e29b-41d4-a716-446655440000",
        })

        assert isinstance(event, AgentEvent)
    """

    # ------------------------------------------------------------------
    # Primary conversion method
    # ------------------------------------------------------------------

    def from_dict(self, event_dict: dict[str, Any]) -> AgentEvent:
        """
        Normalize an arbitrary dictionary into a validated :class:`AgentEvent`.

        Processing steps:

        1. Shallow-copy the dict to avoid mutating the caller's data.
        2. Apply :data:`_FIELD_ALIASES` to rename non-canonical keys.
        3. Wrap scalar alias values in a ``{"value": …}`` dict when the
           canonical field expects a ``dict`` (e.g. ``prompt`` → ``input_data``).
        4. Normalize the ``timestamp`` field to UTC-aware datetime.
        5. Coerce UUID string fields (``trace_id``, ``event_id``, ``parent_id``).
        6. Coerce ``event_type``, ``status``, ``observation_level`` to their
           enum types, applying safe defaults when values are missing/invalid.
        7. Build and return a frozen :class:`AgentEvent`.

        Args:
            event_dict: Raw event data in any supported shape.

        Returns:
            A fully validated :class:`AgentEvent` instance.
        """
        payload: dict[str, Any] = dict(event_dict)

        # ---- 1. Apply field aliases ------------------------------------------
        # Aliases that map to dict-typed fields need special handling: if the
        # incoming value is a plain string/int, wrap it in {"value": …}.
        _DICT_FIELDS = {"input_data", "output_data"}

        for alias, canonical in _FIELD_ALIASES.items():
            if alias in payload and canonical not in payload:
                raw = payload.pop(alias)
                if canonical in _DICT_FIELDS and not isinstance(raw, dict):
                    payload[canonical] = {"value": raw}
                else:
                    payload[canonical] = raw
            elif alias in payload:
                # canonical already present — discard the alias silently
                payload.pop(alias)

        # ---- 2. Timestamp normalization --------------------------------------
        if "timestamp" in payload:
            payload["timestamp"] = _normalize_timestamp(payload["timestamp"])
        else:
            payload["timestamp"] = datetime.now(timezone.utc)

        # ---- 3. UUID coercion ------------------------------------------------
        for uuid_field in ("trace_id", "event_id", "parent_id"):
            raw_val = payload.get(uuid_field)
            if raw_val is None:
                continue
            if not isinstance(raw_val, UUID):
                try:
                    payload[uuid_field] = UUID(str(raw_val))
                except (ValueError, AttributeError):
                    warnings.warn(
                        f"AgentLens EventNormalizer: invalid UUID for '{uuid_field}': "
                        f"{raw_val!r}. Generating a new one.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    if uuid_field == "trace_id":
                        payload[uuid_field] = uuid4()
                    else:
                        payload.pop(uuid_field, None)

        # Ensure trace_id exists (required field on AgentEvent).
        if "trace_id" not in payload or payload.get("trace_id") is None:
            payload["trace_id"] = uuid4()

        # ---- 4. Enum coercion ------------------------------------------------
        if "event_type" in payload:
            raw_et = payload["event_type"]
            if not isinstance(raw_et, EventType):
                try:
                    payload["event_type"] = EventType(str(raw_et).lower())
                except ValueError:
                    warnings.warn(
                        f"AgentLens EventNormalizer: unknown event_type {raw_et!r}. "
                        "Defaulting to 'llm_call'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    payload["event_type"] = EventType.LLM_CALL
        else:
            payload["event_type"] = EventType.LLM_CALL

        if "status" in payload:
            raw_st = payload["status"]
            if not isinstance(raw_st, EventStatus):
                try:
                    payload["status"] = EventStatus(str(raw_st).lower())
                except ValueError:
                    payload["status"] = EventStatus.SUCCESS

        if "observation_level" in payload:
            raw_ol = payload["observation_level"]
            if not isinstance(raw_ol, ObservationLevel):
                try:
                    payload["observation_level"] = ObservationLevel(str(raw_ol).lower())
                except ValueError:
                    payload["observation_level"] = ObservationLevel.FULL

        # ---- 5. agent_id default --------------------------------------------
        if "agent_id" not in payload or not payload["agent_id"]:
            payload["agent_id"] = "unknown"

        # ---- 6. Build AgentEvent --------------------------------------------
        try:
            return AgentEvent(**payload)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"AgentLens EventNormalizer: failed to build AgentEvent: {exc!r}. "
                "Returning a minimal fallback event.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Last-resort fallback — strip all non-essential fields.
            return AgentEvent(
                trace_id=payload.get("trace_id") or uuid4(),
                agent_id=payload.get("agent_id") or "unknown",
                event_type=payload.get("event_type") or EventType.LLM_CALL,
            )

    # ------------------------------------------------------------------
    # Framework-specific stubs
    # ------------------------------------------------------------------

    def from_langchain_event(self, event: Any) -> AgentEvent:
        """
        Normalize a LangChain callback event into an :class:`AgentEvent`.

        .. note::
            **Stub — not yet implemented.**

            This method is reserved for the LangChain adapter sprint.
            Currently raises :exc:`NotImplementedError`.

        Args:
            event: A LangChain callback event object (e.g. from
                ``BaseCallbackHandler``).

        Returns:
            Normalized :class:`AgentEvent`.

        Raises:
            NotImplementedError: Until the LangChain adapter is implemented.
        """
        raise NotImplementedError(
            "LangChain event normalization is not yet implemented.  "
            "This stub is reserved for the LangChain adapter sprint."
        )

    def from_llamaindex_event(self, event: Any) -> AgentEvent:
        """
        Normalize a LlamaIndex event into an :class:`AgentEvent`.

        .. note::
            **Stub — not yet implemented.**

            This method is reserved for the LlamaIndex adapter sprint.
            Currently raises :exc:`NotImplementedError`.

        Args:
            event: A LlamaIndex CBEvent or similar object.

        Returns:
            Normalized :class:`AgentEvent`.

        Raises:
            NotImplementedError: Until the LlamaIndex adapter is implemented.
        """
        raise NotImplementedError(
            "LlamaIndex event normalization is not yet implemented.  "
            "This stub is reserved for the LlamaIndex adapter sprint."
        )
