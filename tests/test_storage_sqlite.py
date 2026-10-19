"""
Reliability tests for the SQLite storage backend (Phase 9, Member 3).

These tests treat the backend as a black box and stress the properties that
matter for telemetry storage:

- **Fidelity**    — round-trips preserve UUIDs, datetimes, JSON and unicode.
- **Ordering**    — events come back in ``timestamp`` order.
- **Concurrency** — many writers at once never hit ``database is locked``.
- **Durability**  — data survives closing and reopening the same file.
- **Migrations**  — the schema applies exactly once and upgrades cleanly.
- **Parity**      — SQLite behaves like the in-memory store.
- **Fail-open**   — a broken backend never crashes the agent.

All tests are async (pytest-asyncio, ``asyncio_mode = "auto"``).
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import pytest

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    ObservationLevel,
    Trace,
)
from agentlens.sdk_collector import EventCollector
from agentlens.storage import db as db_module
from agentlens.storage.db import Database, _run_migrations, close_database, get_database, init_db
from agentlens.storage.memory import InMemoryEventStore, InMemoryTraceStore
from agentlens.storage.sqlite import SqliteEventStore, SqliteTraceStore

pytestmark = pytest.mark.unit


BASE = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


# ===========================================================================
# Fixtures & helpers
# ===========================================================================


@pytest.fixture
async def database(tmp_path) -> AsyncIterator[Database]:
    """A migrated SQLite database in a temporary file."""
    db = Database(tmp_path / "agentlens.db")
    await db._open()
    await _run_migrations(db)
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
async def event_store(database: Database) -> SqliteEventStore:
    return SqliteEventStore(database)


@pytest.fixture
async def trace_store(database: Database) -> SqliteTraceStore:
    return SqliteTraceStore(database)


def _event(
    trace_id: UUID,
    *,
    ts: datetime = BASE,
    agent_id: str = "agent-a",
    status: EventStatus = EventStatus.SUCCESS,
    event_type: EventType = EventType.LLM_CALL,
    parent_id: UUID | None = None,
    **kwargs: Any,
) -> AgentEvent:
    """Build an event with an explicit, deterministic timestamp."""
    return AgentEvent(
        trace_id=trace_id,
        parent_id=parent_id,
        agent_id=agent_id,
        event_type=event_type,
        timestamp=ts,
        status=status,
        **kwargs,
    )


def _trace(
    trace_id: UUID,
    *,
    start_time: datetime = BASE,
    agent_id: str = "agent-a",
    status: EventStatus = EventStatus.SUCCESS,
    events: list[AgentEvent] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trace:
    """Build a trace; ``event_count``/``total_latency_ms`` mirror the events."""
    events = events or []
    return Trace(
        trace_id=trace_id,
        agent_id=agent_id,
        status=status,
        start_time=start_time,
        end_time=start_time + timedelta(seconds=1),
        event_count=len(events),
        total_latency_ms=sum(e.latency_ms or 0.0 for e in events),
        events=events,
        metadata=metadata,
    )


async def _count(db: Database, table: str, where: str = "", params: tuple = ()) -> int:
    row = await db.fetch_one(f"SELECT COUNT(*) AS cnt FROM {table} {where}", params)
    return int(row["cnt"])


# ===========================================================================
# Connection configuration
# ===========================================================================


class TestConnectionConfig:
    async def test_wal_mode_is_enabled(self, database: Database) -> None:
        row = await database.fetch_one("PRAGMA journal_mode")
        assert row[0].lower() == "wal"

    async def test_synchronous_is_normal(self, database: Database) -> None:
        row = await database.fetch_one("PRAGMA synchronous")
        # NORMAL == 1
        assert int(row[0]) == 1

    async def test_busy_timeout_is_set(self, database: Database) -> None:
        row = await database.fetch_one("PRAGMA busy_timeout")
        assert int(row[0]) == 5000


# ===========================================================================
# Round-trip fidelity
# ===========================================================================


class TestRoundTrip:
    async def test_event_round_trip_fidelity(self, event_store) -> None:
        trace_id = uuid4()
        parent_id = uuid4()
        ts = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
        event = _event(
            trace_id,
            ts=ts,
            parent_id=parent_id,
            input_data={"nested": {"a": [1, 2, 3]}, "unicode": "héllo — 世界", "none": None},
            output_data={"big": "x" * 5000},
            metadata={"key": "value", "n": 7},
            latency_ms=12.5,
            importance_score=0.75,
            status=EventStatus.PARTIAL,
            observation_level=ObservationLevel.COMPRESSED,
        )

        await event_store.save_event(event)
        loaded = await event_store.get_event(event.event_id)

        assert loaded == event

    async def test_get_missing_event_raises_key_error(self, event_store) -> None:
        with pytest.raises(KeyError):
            await event_store.get_event(uuid4())

    async def test_upsert_on_same_event_id(self, event_store, database) -> None:
        trace_id = uuid4()
        event = _event(trace_id, output_data={"v": 1})
        await event_store.save_event(event)
        await event_store.save_event(event)  # same id, upsert
        assert await _count(database, "events") == 1

    async def test_get_trace_events_ordered_by_timestamp(self, event_store) -> None:
        trace_id = uuid4()
        # Insert deliberately out of order.
        timestamps = [BASE + timedelta(seconds=s) for s in (3, 1, 2, 0)]
        for ts in timestamps:
            await event_store.save_event(_event(trace_id, ts=ts))

        loaded = await event_store.get_trace_events(trace_id)
        assert [e.timestamp for e in loaded] == sorted(timestamps)

    async def test_trace_round_trip(self, trace_store, event_store) -> None:
        trace_id = uuid4()
        events = [_event(trace_id, ts=BASE + timedelta(seconds=i)) for i in range(3)]
        for e in events:
            await event_store.save_event(e)
        trace = _trace(trace_id, events=events, metadata={"agent": "x"})

        await trace_store.save_trace(trace)
        loaded = await trace_store.get_trace(trace_id)

        assert loaded.trace_id == trace_id
        assert loaded.status == EventStatus.SUCCESS
        assert loaded.event_count == 3
        assert loaded.metadata == {"agent": "x"}
        assert [e.event_id for e in loaded.events] == [e.event_id for e in events]

    async def test_get_missing_trace_raises_key_error(self, trace_store) -> None:
        with pytest.raises(KeyError):
            await trace_store.get_trace(uuid4())

    async def test_bulk_insert_matches_individual_saves(self, event_store, database) -> None:
        trace_id = uuid4()
        events = [_event(trace_id, ts=BASE + timedelta(milliseconds=i)) for i in range(50)]
        await event_store.save_events_bulk(events)

        assert await _count(database, "events", "WHERE trace_id = ?", (str(trace_id),)) == 50
        loaded = await event_store.get_trace_events(trace_id)
        assert {e.event_id for e in loaded} == {e.event_id for e in events}

    async def test_bulk_insert_empty_is_noop(self, event_store, database) -> None:
        await event_store.save_events_bulk([])
        assert await _count(database, "events") == 0


# ===========================================================================
# Keyset pagination
# ===========================================================================


class TestPagination:
    async def test_pages_cover_all_rows_without_duplicates(self, trace_store) -> None:
        # Equal start_time for every row forces the trace_id tie-breaker to work.
        ids = sorted(str(uuid4()) for _ in range(25))
        for tid in ids:
            await trace_store.save_trace(_trace(UUID(tid), start_time=BASE))

        seen: list[str] = []
        cursor = None
        pages = 0
        while True:
            page, cursor = await trace_store.list_trace_page(limit=10, cursor=cursor)
            seen.extend(str(t.trace_id) for t in page)
            pages += 1
            if cursor is None:
                break
            assert pages < 10, "cursor never terminated"

        assert pages == 3
        assert len(seen) == len(set(seen)) == 25
        assert sorted(seen) == ids

    async def test_exact_multiple_has_no_extra_page(self, trace_store) -> None:
        for _ in range(4):
            await trace_store.save_trace(_trace(uuid4(), start_time=BASE))
        page, cursor = await trace_store.list_trace_page(limit=2)
        assert len(page) == 2
        page2, cursor2 = await trace_store.list_trace_page(limit=2, cursor=cursor)
        assert len(page2) == 2
        assert cursor2 is None

    async def test_first_page_order_is_start_time_desc(self, trace_store) -> None:
        for i in range(5):
            await trace_store.save_trace(
                _trace(uuid4(), start_time=BASE + timedelta(seconds=i))
            )
        page, _ = await trace_store.list_trace_page(limit=5)
        starts = [t.start_time for t in page]
        assert starts == sorted(starts, reverse=True)

    async def test_list_page_returns_summaries_only(self, trace_store, event_store) -> None:
        trace_id = uuid4()
        await event_store.save_event(_event(trace_id))
        await trace_store.save_trace(_trace(trace_id))

        page, _ = await trace_store.list_trace_page(limit=5)
        assert page[0].events == []  # summaries do not hydrate events


# ===========================================================================
# Filtering & counts
# ===========================================================================


class TestFilters:
    async def test_status_filter(self, trace_store) -> None:
        await trace_store.save_trace(_trace(uuid4(), status=EventStatus.SUCCESS))
        await trace_store.save_trace(_trace(uuid4(), status=EventStatus.FAILURE))
        await trace_store.save_trace(_trace(uuid4(), status=EventStatus.FAILURE))

        assert await trace_store.count_traces(status=EventStatus.FAILURE) == 2
        page, _ = await trace_store.list_trace_page(status=EventStatus.FAILURE)
        assert len(page) == 2

    async def test_agent_filter(self, trace_store) -> None:
        await trace_store.save_trace(_trace(uuid4(), agent_id="a"))
        await trace_store.save_trace(_trace(uuid4(), agent_id="b"))
        assert await trace_store.count_traces(agent_id="a") == 1

    async def test_start_date_is_inclusive(self, trace_store) -> None:
        await trace_store.save_trace(_trace(uuid4(), start_time=BASE))
        assert await trace_store.count_traces(start_date=BASE) == 1
        assert await trace_store.count_traces(start_date=BASE + timedelta(seconds=1)) == 0

    async def test_end_date_is_inclusive(self, trace_store) -> None:
        await trace_store.save_trace(_trace(uuid4(), start_time=BASE))
        assert await trace_store.count_traces(end_date=BASE) == 1
        assert await trace_store.count_traces(end_date=BASE - timedelta(seconds=1)) == 0

    async def test_count_matches_full_page_iteration(self, trace_store) -> None:
        for i in range(7):
            await trace_store.save_trace(
                _trace(
                    uuid4(),
                    start_time=BASE + timedelta(seconds=i),
                    status=EventStatus.SUCCESS if i % 2 == 0 else EventStatus.FAILURE,
                )
            )
        total = await trace_store.count_traces(status=EventStatus.SUCCESS)
        seen = 0
        cursor = None
        while True:
            page, cursor = await trace_store.list_trace_page(
                limit=2, status=EventStatus.SUCCESS, cursor=cursor
            )
            seen += len(page)
            if cursor is None:
                break
        assert seen == total == 4


# ===========================================================================
# Concurrency
# ===========================================================================


class TestConcurrency:
    async def test_concurrent_writes_are_safe(self, event_store, trace_store, database) -> None:
        n_traces = 40
        events_per_trace = 5

        async def write_one(i: int) -> None:
            trace_id = uuid4()
            events = [
                _event(trace_id, ts=BASE + timedelta(microseconds=j))
                for j in range(events_per_trace)
            ]
            for event in events:
                await event_store.save_event(event)
            await trace_store.save_trace(_trace(trace_id, events=events))

        results = await asyncio.gather(
            *(write_one(i) for i in range(n_traces)), return_exceptions=True
        )

        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, f"concurrent writes raised: {errors!r}"

        assert await _count(database, "events") == n_traces * events_per_trace
        assert await _count(database, "traces") == n_traces

    async def test_mixed_read_write_under_concurrency(self, event_store, database) -> None:
        trace_id = uuid4()
        await event_store.save_event(_event(trace_id))

        async def writer(i: int) -> None:
            await event_store.save_event(_event(trace_id, ts=BASE + timedelta(seconds=i)))

        async def reader() -> None:
            await event_store.get_trace_events(trace_id)

        await asyncio.gather(*(writer(i) for i in range(20)), *(reader() for _ in range(20)))
        assert await _count(database, "events") == 21


# ===========================================================================
# Durability across reopen
# ===========================================================================


class TestDurability:
    async def test_data_survives_close_and_reopen(self, tmp_path, event_store, trace_store) -> None:
        trace_id = uuid4()
        events = [_event(trace_id, ts=BASE + timedelta(seconds=i)) for i in range(3)]
        for e in events:
            await event_store.save_event(e)
        await trace_store.save_trace(_trace(trace_id, events=events))
        db_path = event_store._db._path

        # Close the original database entirely.
        await event_store._db.close()

        reopened = Database(db_path)
        await reopened._open()
        try:
            await _run_migrations(reopened)
            new_event_store = SqliteEventStore(reopened)
            new_trace_store = SqliteTraceStore(reopened)

            loaded_events = await new_event_store.get_trace_events(trace_id)
            assert [e.event_id for e in loaded_events] == [e.event_id for e in events]

            loaded_trace = await new_trace_store.get_trace(trace_id)
            assert loaded_trace.event_count == 3
            assert len(loaded_trace.events) == 3
        finally:
            await reopened.close()

    async def test_writes_after_close_fail(self, tmp_path) -> None:
        db = Database(tmp_path / "x.db")
        await db._open()
        await _run_migrations(db)
        store = SqliteEventStore(db)
        await db.close()

        # A closed database is unusable: the write connection is gone.
        with pytest.raises((AssertionError, sqlite3.ProgrammingError)):
            await store.save_event(_event(uuid4()))


# ===========================================================================
# Migrations
# ===========================================================================


class TestMigrations:
    async def test_fresh_database_records_version_1(self, database) -> None:
        rows = await database.fetch_all("SELECT version FROM schema_migrations")
        assert {r["version"] for r in rows} == {1}

    async def test_migrations_are_idempotent(self, database) -> None:
        # Re-running must apply nothing and must not raise.
        await _run_migrations(database)
        await _run_migrations(database)
        assert await _count(database, "schema_migrations") == 1

    async def test_pending_migration_is_applied(self, database, tmp_path, monkeypatch) -> None:
        upgrade_dir = tmp_path / "migrations"
        upgrade_dir.mkdir()
        (upgrade_dir / "0002_add_note_table.sql").write_text(
            "CREATE TABLE note (id INTEGER PRIMARY KEY, body TEXT);",
            encoding="utf-8",
        )
        monkeypatch.setattr(db_module, "_migrations_dir", lambda: upgrade_dir)

        await _run_migrations(database)

        rows = await database.fetch_all("SELECT version FROM schema_migrations")
        assert {r["version"] for r in rows} == {1, 2}
        # The new table exists.
        await database.execute_write("INSERT INTO note (body) VALUES (?)", ("hi",))
        assert await _count(database, "note") == 1

    async def test_applied_migration_is_not_reapplied(self, database, tmp_path, monkeypatch) -> None:
        upgrade_dir = tmp_path / "migrations"
        upgrade_dir.mkdir()
        # A migration file with the *same* version as an applied one must be skipped.
        (upgrade_dir / "0001_evil.sql").write_text(
            "DROP TABLE events;", encoding="utf-8"
        )
        monkeypatch.setattr(db_module, "_migrations_dir", lambda: upgrade_dir)

        await _run_migrations(database)

        # events table still exists → the applied 0001 was not re-run.
        assert await _count(database, "events") == 0


# ===========================================================================
# Failure handling / fail-open
# ===========================================================================


class TestFailureHandling:
    async def test_unwritable_path_raises(self, tmp_path) -> None:
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x", encoding="utf-8")

        db = Database(blocker / "sub" / "agentlens.db")
        with pytest.raises((sqlite3.OperationalError, sqlite3.DatabaseError, OSError)):
            await db._open()

    async def test_init_db_failure_leaves_no_singleton(self, tmp_path) -> None:
        await close_database()
        blocker = tmp_path / "not_a_dir2"
        blocker.write_text("x", encoding="utf-8")

        with pytest.raises((sqlite3.OperationalError, sqlite3.DatabaseError, OSError)):
            await init_db(str(blocker / "sub" / "agentlens.db"))

        with pytest.raises(RuntimeError):
            get_database()

    async def test_collector_is_fail_open_with_broken_store(self, tmp_path) -> None:
        db = Database(tmp_path / "broken.db")
        await db._open()
        await _run_migrations(db)
        store = SqliteEventStore(db)
        await db.close()  # every subsequent save fails

        collector = EventCollector(storage_backend=store)
        await collector.start_trace(agent_id="resilient-agent")

        with pytest.warns(RuntimeWarning):
            event = await collector.record_event(
                event_type=EventType.LLM_CALL, input_data={"prompt": "hi"}
            )

        assert event is not None  # telemetry failure never propagates


# ===========================================================================
# Parity with the in-memory backend
# ===========================================================================


async def _scenario(event_store, trace_store, ids: dict[str, UUID]) -> dict[str, Any]:
    """Run an identical workload and return a comparable summary.

    ``ids`` supplies fixed UUIDs so two backends can be compared exactly.
    """
    t1, t2 = ids["t1"], ids["t2"]
    e1 = _event(
        t1,
        ts=BASE,
        status=EventStatus.SUCCESS,
        input_data={"p": 1},
        latency_ms=1.0,
        event_id=ids["e1"],
    )
    e2 = _event(
        t1,
        ts=BASE + timedelta(seconds=1),
        status=EventStatus.FAILURE,
        output_data={"err": "boom"},
        latency_ms=2.0,
        event_id=ids["e2"],
    )
    e3 = _event(
        t2,
        ts=BASE + timedelta(seconds=2),
        agent_id="agent-b",
        latency_ms=3.0,
        event_id=ids["e3"],
    )
    for e in (e1, e2, e3):
        await event_store.save_event(e)

    await trace_store.save_trace(
        _trace(t1, start_time=BASE, status=EventStatus.PARTIAL, events=[e1, e2])
    )
    await trace_store.save_trace(
        _trace(t2, start_time=BASE + timedelta(seconds=2), agent_id="agent-b", events=[e3])
    )

    loaded = await trace_store.get_trace(t1)
    page, cursor = await trace_store.list_trace_page(limit=1)

    return {
        "t1_event_ids": [str(e.event_id) for e in await event_store.get_trace_events(t1)],
        "distinct_trace_ids": sorted(str(x) for x in await event_store.list_traces()),
        "t1_status": loaded.status.value,
        "t1_event_count": loaded.event_count,
        "t1_hydrated_events": len(loaded.events),
        "count_total": await trace_store.count_traces(),
        "count_failure": await trace_store.count_traces(status=EventStatus.FAILURE),
        "count_agent_b": await trace_store.count_traces(agent_id="agent-b"),
        "page1_ids": [str(t.trace_id) for t in page],
        "has_cursor": cursor is not None,
    }


class TestParityWithMemory:
    @pytest.mark.integration
    async def test_sqlite_matches_in_memory(self, event_store, trace_store) -> None:
        ids = {k: uuid4() for k in ("t1", "t2", "e1", "e2", "e3")}

        sqlite_summary = await _scenario(event_store, trace_store, ids)
        memory_summary = await _scenario(InMemoryEventStore(), InMemoryTraceStore(), ids)

        assert sqlite_summary == memory_summary
