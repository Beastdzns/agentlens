"""
SQLite-backed storage implementations for AgentLens.

Classes
-------
- :class:`SqliteEventStore` — implements :class:`~agentlens.storage.base.EventStore`
- :class:`SqliteTraceStore` — implements :class:`~agentlens.storage.base.TraceStore`

Both classes rely on the :class:`~agentlens.storage.db.Database` singleton
(initialised via :func:`~agentlens.storage.db.init_db`) for all I/O.

Serialisation
-------------
- Pydantic models → ``model_dump(mode="json")`` → ``json.dumps`` → TEXT column.
- Read back with ``AgentEvent.model_validate(json.loads(...))``.
- Timestamps stored as UTC ISO-8601 with microsecond precision (TEXT-sortable).

Pagination
----------
:meth:`SqliteTraceStore.list_trace_page` uses **keyset** (cursor) pagination:

    WHERE (start_time, trace_id) < (:st, :tid)
    ORDER BY start_time DESC, trace_id DESC
    LIMIT :limit + 1

Cursor is ``base64("<start_time>|<trace_id>")``, encoded/decoded transparently.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from agentlens.events import AgentEvent, EventStatus, Trace
from agentlens.storage.base import EventStore, TraceStore
from agentlens.storage.db import Database

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────


def _dt_to_str(dt: datetime) -> str:
    """Convert a datetime to a fixed-precision UTC ISO-8601 string (TEXT-sortable)."""
    # Ensure UTC, then format with exactly 6 decimal places for sort stability.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _str_to_dt(s: str) -> datetime:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _encode_cursor(start_time: str, trace_id: str) -> str:
    """Encode keyset cursor as base64('<start_time>|<trace_id>')."""
    raw = f"{start_time}|{trace_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[str, str]:
    """Decode a cursor produced by :func:`_encode_cursor`."""
    raw = base64.urlsafe_b64decode(cursor.encode()).decode()
    start_time, trace_id = raw.rsplit("|", 1)
    return start_time, trace_id


# ── SqliteEventStore ──────────────────────────────────────────────────────────


class SqliteEventStore(EventStore):
    """
    SQLite-backed implementation of :class:`~agentlens.storage.base.EventStore`.

    Args:
        db: An open :class:`~agentlens.storage.db.Database` instance.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ── write ──────────────────────────────────────────────────────────────

    async def save_event(self, event: AgentEvent) -> None:
        """
        Persist a single :class:`~agentlens.events.AgentEvent` (upsert).

        JSON-encodes ``input_data``, ``output_data``, and ``metadata``.
        Timestamps are stored as TEXT in UTC ISO-8601 with microsecond
        precision so they sort correctly as strings.
        """
        data = event.model_dump(mode="json")
        await self._db.execute_write(
            """
            INSERT OR REPLACE INTO events
                (event_id, trace_id, parent_id, agent_id, event_type, status,
                 timestamp, input_data, output_data, metadata,
                 latency_ms, importance_score, observation_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event.event_id),
                str(event.trace_id),
                str(event.parent_id) if event.parent_id else None,
                event.agent_id,
                data["event_type"],
                data["status"],
                _dt_to_str(event.timestamp),
                json.dumps(data["input_data"]) if data["input_data"] is not None else None,
                json.dumps(data["output_data"]) if data["output_data"] is not None else None,
                json.dumps(data["metadata"]) if data["metadata"] is not None else None,
                event.latency_ms,
                event.importance_score,
                data["observation_level"],
            ),
        )

    async def save_events_bulk(self, events: list[AgentEvent]) -> None:
        """
        Persist multiple events in a single ``executemany`` transaction.

        This is significantly faster than calling :meth:`save_event` in a loop
        when ingesting large batches.

        Args:
            events: List of :class:`~agentlens.events.AgentEvent` to persist.
        """
        if not events:
            return

        rows = []
        for event in events:
            data = event.model_dump(mode="json")
            rows.append((
                str(event.event_id),
                str(event.trace_id),
                str(event.parent_id) if event.parent_id else None,
                event.agent_id,
                data["event_type"],
                data["status"],
                _dt_to_str(event.timestamp),
                json.dumps(data["input_data"]) if data["input_data"] is not None else None,
                json.dumps(data["output_data"]) if data["output_data"] is not None else None,
                json.dumps(data["metadata"]) if data["metadata"] is not None else None,
                event.latency_ms,
                event.importance_score,
                data["observation_level"],
            ))

        await self._db.executemany_write(
            """
            INSERT OR REPLACE INTO events
                (event_id, trace_id, parent_id, agent_id, event_type, status,
                 timestamp, input_data, output_data, metadata,
                 latency_ms, importance_score, observation_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    # ── read ───────────────────────────────────────────────────────────────

    async def get_event(self, event_id: UUID) -> AgentEvent:
        """
        Retrieve a single event by ``event_id``.

        Raises:
            KeyError: If no event with ``event_id`` exists.
        """
        row = await self._db.fetch_one(
            "SELECT * FROM events WHERE event_id = ?", (str(event_id),)
        )
        if row is None:
            raise KeyError(f"Event {event_id!r} not found in store.")
        return _row_to_event(row)

    async def get_trace_events(self, trace_id: UUID) -> list[AgentEvent]:
        """
        Return all events for a trace ordered by ``timestamp`` ASC, then ``event_id`` ASC.

        Returns an empty list when the trace has no events.
        """
        rows = await self._db.fetch_all(
            """
            SELECT * FROM events
            WHERE  trace_id = ?
            ORDER  BY timestamp ASC, event_id ASC
            """,
            (str(trace_id),),
        )
        return [_row_to_event(r) for r in rows]

    # ── delete ─────────────────────────────────────────────────────────────

    async def delete_trace(self, trace_id: UUID) -> None:
        """Delete all events belonging to ``trace_id``. No-op if trace has no events."""
        await self._db.execute_write(
            "DELETE FROM events WHERE trace_id = ?", (str(trace_id),)
        )

    # ── list ───────────────────────────────────────────────────────────────

    async def list_traces(self, limit: int = 100) -> list[UUID]:
        """Return distinct trace UUIDs that have at least one event (most recent first)."""
        rows = await self._db.fetch_all(
            """
            SELECT DISTINCT trace_id, MAX(timestamp) AS last_ts
            FROM   events
            GROUP  BY trace_id
            ORDER  BY last_ts DESC
            LIMIT  ?
            """,
            (limit,),
        )
        return [UUID(row["trace_id"]) for row in rows]


# ── SqliteTraceStore ──────────────────────────────────────────────────────────


class SqliteTraceStore(TraceStore):
    """
    SQLite-backed implementation of :class:`~agentlens.storage.base.TraceStore`.

    Args:
        db: An open :class:`~agentlens.storage.db.Database` instance.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ── write ──────────────────────────────────────────────────────────────

    async def save_trace(self, trace: Trace) -> None:
        """Persist a :class:`~agentlens.events.Trace` object (upsert semantics)."""
        data = trace.model_dump(mode="json")
        await self._db.execute_write(
            """
            INSERT OR REPLACE INTO traces
                (trace_id, agent_id, status, start_time, end_time,
                 event_count, total_latency_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(trace.trace_id),
                trace.agent_id,
                data["status"],
                _dt_to_str(trace.start_time),
                _dt_to_str(trace.end_time) if trace.end_time else None,
                trace.event_count,
                trace.total_latency_ms,
                json.dumps(data["metadata"]) if data["metadata"] is not None else None,
            ),
        )

    async def update_trace(self, trace_id: UUID, updates: dict[str, Any]) -> None:
        """
        Apply a partial update to an existing trace.

        Only recognised field names from :class:`~agentlens.events.Trace`
        are applied; unknown keys are silently ignored.

        Raises:
            KeyError: If no trace with ``trace_id`` exists.
        """
        row = await self._db.fetch_one(
            "SELECT * FROM traces WHERE trace_id = ?", (str(trace_id),)
        )
        if row is None:
            raise KeyError(f"Trace {trace_id!r} not found in store.")

        trace = _row_to_trace(row)
        current = trace.model_dump()
        for key, value in updates.items():
            if key in current:
                current[key] = value

        updated = Trace(**current)
        await self.save_trace(updated)

    # ── read ───────────────────────────────────────────────────────────────

    async def get_trace(self, trace_id: UUID) -> Trace:
        """
        Retrieve a single trace by UUID.

        Raises:
            KeyError: If no trace with ``trace_id`` exists.
        """
        row = await self._db.fetch_one(
            "SELECT * FROM traces WHERE trace_id = ?", (str(trace_id),)
        )
        if row is None:
            raise KeyError(f"Trace {trace_id!r} not found in store.")
        return _row_to_trace(row)

    async def list_traces(
        self,
        limit: int = 100,
        agent_id: Optional[str] = None,
        status: Optional[EventStatus] = None,
    ) -> list[Trace]:
        """
        Return traces ordered by ``start_time`` DESC, optionally filtered.

        Args:
            limit:    Maximum number of traces.
            agent_id: Filter to a specific agent.
            status:   Filter by status.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value if isinstance(status, EventStatus) else status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._db.fetch_all(
            f"SELECT * FROM traces {where} ORDER BY start_time DESC, trace_id DESC LIMIT ?",
            params,
        )
        return [_row_to_trace(r) for r in rows]

    async def list_trace_page(
        self,
        *,
        limit: int = 20,
        agent_id: Optional[str] = None,
        status: Optional[EventStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cursor: Optional[str] = None,
    ) -> tuple[list[Trace], Optional[str]]:
        """
        Return a page of traces using keyset (cursor) pagination.

        The result is always ordered ``start_time DESC, trace_id DESC``.
        The ``cursor`` from a previous call can be passed to retrieve the
        next page.

        Args:
            limit:      Page size (default 20).
            agent_id:   Filter by agent.
            status:     Filter by :class:`~agentlens.events.EventStatus`.
            start_date: Only traces whose ``start_time >= start_date``.
            end_date:   Only traces whose ``start_time <= end_date``.
            cursor:     Opaque continuation token from a previous call.

        Returns:
            ``(traces, next_cursor)`` where ``next_cursor`` is ``None`` when
            there are no more results.

        The returned :class:`~agentlens.events.Trace` objects have their
        ``events`` list empty; only the summary fields are populated.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value if isinstance(status, EventStatus) else status)
        if start_date is not None:
            conditions.append("start_time >= ?")
            params.append(_dt_to_str(start_date))
        if end_date is not None:
            conditions.append("start_time <= ?")
            params.append(_dt_to_str(end_date))

        # Keyset condition using row-value comparison
        if cursor is not None:
            cursor_st, cursor_tid = _decode_cursor(cursor)
            conditions.append("(start_time, trace_id) < (?, ?)")
            params.extend([cursor_st, cursor_tid])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        # Fetch limit+1 to detect whether another page exists
        params.append(limit + 1)

        rows = await self._db.fetch_all(
            f"""
            SELECT * FROM traces
            {where}
            ORDER BY start_time DESC, trace_id DESC
            LIMIT ?
            """,
            params,
        )

        has_next = len(rows) > limit
        page_rows = rows[:limit]
        traces = [_row_to_trace(r) for r in page_rows]

        next_cursor: Optional[str] = None
        if has_next and traces:
            last = traces[-1]
            next_cursor = _encode_cursor(_dt_to_str(last.start_time), str(last.trace_id))

        return traces, next_cursor

    async def count_traces(
        self,
        *,
        agent_id: Optional[str] = None,
        status: Optional[EventStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Return the total number of traces matching the given filters.

        Args:
            agent_id:   Filter by agent ID.
            status:     Filter by status.
            start_date: Minimum ``start_time`` (inclusive).
            end_date:   Maximum ``start_time`` (inclusive).

        Returns:
            Integer count.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value if isinstance(status, EventStatus) else status)
        if start_date is not None:
            conditions.append("start_time >= ?")
            params.append(_dt_to_str(start_date))
        if end_date is not None:
            conditions.append("start_time <= ?")
            params.append(_dt_to_str(end_date))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        row = await self._db.fetch_one(f"SELECT COUNT(*) AS cnt FROM traces {where}", params)
        return int(row["cnt"]) if row else 0


# ── row-to-model helpers ──────────────────────────────────────────────────────


def _row_to_event(row: Any) -> AgentEvent:
    """Deserialise a database row into an :class:`~agentlens.events.AgentEvent`."""
    d: dict[str, Any] = dict(row)

    # JSON columns
    for col in ("input_data", "output_data", "metadata"):
        if d.get(col) is not None:
            d[col] = json.loads(d[col])

    # Validate via Pydantic (handles enum coercion, UUID parsing, etc.)
    return AgentEvent.model_validate(d)


def _row_to_trace(row: Any) -> Trace:
    """Deserialise a database row into a :class:`~agentlens.events.Trace` summary."""
    d: dict[str, Any] = dict(row)

    # JSON columns
    if d.get("metadata") is not None:
        d["metadata"] = json.loads(d["metadata"])

    # Summaries never carry embedded events
    d["events"] = []

    return Trace.model_validate(d)
