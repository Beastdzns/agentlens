"""
Unit tests for the storage reliability layer (Phase 9, Member 3).

Unlike ``test_storage_sqlite.py`` (integration-style, real SQLite files), these
tests isolate the individual units touched by the reliability work:

- datetime/cursor serialisation helpers;
- row → model mapping;
- store control flow, using a stub ``Database`` (no I/O);
- per-connection PRAGMA setup;
- the database singleton lifecycle;
- the migration runner's file/version handling;
- backend selection (``AGENTLENS_STORAGE``).

All tests are async unless they exercise pure functions
(pytest-asyncio, ``asyncio_mode = "auto"``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

import agentlens.web_api as web_api
from agentlens.events import EventStatus, EventType
from agentlens.storage import db as db_module
from agentlens.storage.db import (
    Database,
    _applied_versions,
    _configure_connection,
    _run_migrations,
    close_database,
    get_database,
    init_db,
)
from agentlens.storage.sqlite import (
    SqliteTraceStore,
    _decode_cursor,
    _dt_to_str,
    _encode_cursor,
    _row_to_event,
    _row_to_trace,
    _str_to_dt,
)

pytestmark = pytest.mark.unit

UTC = timezone.utc
ISO = "2026-05-01T12:00:00.000000+00:00"


# ============================================================================
# Test doubles
# ============================================================================


class FakeDatabase:
    """A ``Database`` stand-in that records calls and returns canned rows."""

    def __init__(self, *, one=None, all=None):  # noqa: A002 - mirrors SQL naming
        self._one = one
        self._all = list(all or [])
        self.calls: list[tuple[str, str, object]] = []

    async def fetch_one(self, sql, params=()):  # noqa: ANN001
        self.calls.append(("fetch_one", sql, tuple(params)))
        return self._one

    async def fetch_all(self, sql, params=()):  # noqa: ANN001
        self.calls.append(("fetch_all", sql, tuple(params)))
        return self._all

    async def execute_write(self, sql, params=()):  # noqa: ANN001
        self.calls.append(("execute_write", sql, tuple(params)))
        return None

    async def executemany_write(self, sql, params_seq):  # noqa: ANN001
        rows = list(params_seq)
        self.calls.append(("executemany_write", sql, len(rows)))


class RecordingConnection:
    """Minimal async connection that records executed PRAGMA statements."""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.row_factory = None

    async def execute(self, sql, params=()):  # noqa: ANN001
        self.executed.append(sql)
        return None


# ── row builders ────────────────────────────────────────────────────────────


def _event_row(trace_id: UUID, *, ts: str = ISO, **over) -> dict:
    row = {
        "event_id": str(uuid4()),
        "trace_id": str(trace_id),
        "parent_id": None,
        "agent_id": "agent-x",
        "event_type": "llm_call",
        "status": "success",
        "timestamp": ts,
        "input_data": None,
        "output_data": None,
        "metadata": None,
        "latency_ms": 1.5,
        "importance_score": 0.5,
        "observation_level": "full",
    }
    row.update(over)
    return row


def _trace_row(*, offset: int = 0, **over) -> dict:
    row = {
        "trace_id": str(uuid4()),
        "agent_id": "agent-a",
        "status": "success",
        "start_time": _dt_to_str(
            datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC) - timedelta(seconds=offset)
        ),
        "end_time": None,
        "event_count": 0,
        "total_latency_ms": 0.0,
        "metadata": None,
    }
    row.update(over)
    return row


# ============================================================================
# Datetime serialisation
# ============================================================================


class TestDatetimeSerialisation:
    def test_aware_utc_uses_fixed_microsecond_precision(self) -> None:
        dt = datetime(2026, 5, 1, 12, 0, 0, 123456, tzinfo=UTC)
        assert _dt_to_str(dt) == "2026-05-01T12:00:00.123456+00:00"

    def test_naive_is_treated_as_utc(self) -> None:
        dt = datetime(2026, 5, 1, 12, 0, 0)
        assert _dt_to_str(dt) == "2026-05-01T12:00:00.000000+00:00"

    def test_non_utc_aware_is_converted_to_utc(self) -> None:
        tz = timezone(timedelta(hours=2))
        dt = datetime(2026, 5, 1, 14, 0, 0, tzinfo=tz)
        assert _dt_to_str(dt) == "2026-05-01T12:00:00.000000+00:00"

    def test_strings_sort_chronologically(self) -> None:
        earlier = _dt_to_str(datetime(2026, 5, 1, 12, 0, 0, 1, tzinfo=UTC))
        later = _dt_to_str(datetime(2026, 5, 1, 12, 0, 0, 2, tzinfo=UTC))
        assert earlier < later

    def test_str_to_dt_round_trips(self) -> None:
        dt = datetime(2026, 5, 1, 12, 0, 0, 654321, tzinfo=UTC)
        assert _str_to_dt(_dt_to_str(dt)) == dt

    def test_str_to_dt_assumes_utc_for_naive_input(self) -> None:
        parsed = _str_to_dt("2026-05-01T12:00:00.000000")
        assert parsed.tzinfo is not None
        assert parsed == datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# ============================================================================
# Cursor codec
# ============================================================================


class TestCursorCodec:
    def test_round_trip(self) -> None:
        trace_id = str(uuid4())
        assert _decode_cursor(_encode_cursor(ISO, trace_id)) == (ISO, trace_id)

    def test_encoding_is_url_safe(self) -> None:
        cursor = _encode_cursor(ISO, str(uuid4()))
        assert "+" not in cursor
        assert "/" not in cursor

    def test_decoder_splits_on_last_separator(self) -> None:
        # start_time may itself contain no '|', but be defensive.
        assert _decode_cursor(_encode_cursor("a|b", "c")) == ("a|b", "c")


# ============================================================================
# Row → model mapping
# ============================================================================


class TestRowMapping:
    def test_event_json_columns_are_decoded(self) -> None:
        row = _event_row(
            uuid4(),
            input_data='{"a": 1}',
            output_data='{"b": [2, 3]}',
            metadata='{"k": "v"}',
        )
        event = _row_to_event(row)

        assert event.input_data == {"a": 1}
        assert event.output_data == {"b": [2, 3]}
        assert event.metadata == {"k": "v"}

    def test_event_none_columns_stay_none(self) -> None:
        event = _row_to_event(_event_row(uuid4()))
        assert event.input_data is None
        assert event.output_data is None
        assert event.metadata is None

    def test_event_types_are_coerced(self) -> None:
        row = _event_row(uuid4())
        event = _row_to_event(row)

        assert event.event_id == UUID(row["event_id"])
        assert event.trace_id == UUID(row["trace_id"])
        assert event.event_type is EventType.LLM_CALL
        assert event.status is EventStatus.SUCCESS

    def test_trace_row_maps_without_embedded_events(self) -> None:
        row = _trace_row(status="failure", metadata='{"x": 1}')
        trace = _row_to_trace(row)

        assert trace.trace_id == UUID(row["trace_id"])
        assert trace.status is EventStatus.FAILURE
        assert trace.metadata == {"x": 1}
        assert trace.events == []


# ============================================================================
# SqliteTraceStore control flow (stub Database)
# ============================================================================


class TestGetTraceHydrationUnit:
    async def test_hydrates_events_from_events_table(self) -> None:
        trace_id = uuid4()
        trace_row = _trace_row()
        trace_row["trace_id"] = str(trace_id)
        event_rows = [_event_row(trace_id), _event_row(trace_id)]
        fake = FakeDatabase(one=trace_row, all=event_rows)

        trace = await SqliteTraceStore(fake).get_trace(trace_id)

        assert [e.event_id for e in trace.events] == [UUID(r["event_id"]) for r in event_rows]
        # A trace lookup followed by exactly one events query.
        assert [c[0] for c in fake.calls] == ["fetch_one", "fetch_all"]

    async def test_missing_trace_raises_without_querying_events(self) -> None:
        fake = FakeDatabase(one=None)

        with pytest.raises(KeyError):
            await SqliteTraceStore(fake).get_trace(uuid4())

        assert [c[0] for c in fake.calls] == ["fetch_one"]


class TestListTracePageUnit:
    async def test_limit_plus_one_is_trimmed_and_cursor_set(self) -> None:
        rows = [_trace_row(offset=i) for i in range(3)]  # limit=2 → 3 fetched
        fake = FakeDatabase(all=rows)

        page, cursor = await SqliteTraceStore(fake).list_trace_page(limit=2)

        assert len(page) == 2
        assert cursor is not None
        start, trace_id = _decode_cursor(cursor)
        assert trace_id == str(page[-1].trace_id)
        assert start == _dt_to_str(page[-1].start_time)

    async def test_last_page_has_no_cursor(self) -> None:
        rows = [_trace_row(offset=i) for i in range(2)]
        fake = FakeDatabase(all=rows)

        page, cursor = await SqliteTraceStore(fake).list_trace_page(limit=2)

        assert len(page) == 2
        assert cursor is None

    async def test_cursor_bounds_are_passed_as_parameters(self) -> None:
        fake = FakeDatabase(all=[])
        cursor = _encode_cursor(ISO, str(uuid4()))

        await SqliteTraceStore(fake).list_trace_page(limit=1, cursor=cursor)

        _, sql, params = fake.calls[0]
        assert params[0] == ISO
        assert "(start_time, trace_id) <" in sql.replace("\n", " ")


# ============================================================================
# Connection configuration
# ============================================================================


class TestConfigureConnectionUnit:
    async def test_pragmas_and_row_factory_are_applied(self) -> None:
        conn = RecordingConnection()
        await _configure_connection(conn)  # type: ignore[arg-type]

        joined = " ".join(conn.executed).lower()
        assert "journal_mode=wal" in joined
        assert "synchronous=normal" in joined
        assert "busy_timeout=5000" in joined
        assert "foreign_keys=on" in joined
        assert conn.row_factory is not None


# ============================================================================
# Migration runner
# ============================================================================


class TestMigrationRunnerUnit:
    async def test_applied_versions_is_empty_without_table(self, tmp_path) -> None:
        db = Database(tmp_path / "bare.db")
        await db._open()
        try:
            assert await _applied_versions(db) == set()
        finally:
            await db.close()

    async def test_missing_migrations_dir_is_a_noop(self, tmp_path, monkeypatch) -> None:
        db = Database(tmp_path / "no_dir.db")
        await db._open()
        try:
            monkeypatch.setattr(db_module, "_migrations_dir", lambda: tmp_path / "absent")
            await _run_migrations(db)  # must not raise
            assert await _applied_versions(db) == set()
        finally:
            await db.close()

    async def test_unversioned_files_are_skipped(self, tmp_path, monkeypatch) -> None:
        mdir = tmp_path / "migrations"
        mdir.mkdir()
        (mdir / "README.md").write_text("not sql", encoding="utf-8")
        (mdir / "notes.sql").write_text(
            "CREATE TABLE should_not_exist (id INTEGER);", encoding="utf-8"
        )
        (mdir / "0007_add_thing.sql").write_text(
            "CREATE TABLE thing (id INTEGER PRIMARY KEY);", encoding="utf-8"
        )
        monkeypatch.setattr(db_module, "_migrations_dir", lambda: mdir)

        db = Database(tmp_path / "mig.db")
        await db._open()
        try:
            await _run_migrations(db)
            versions = {
                r["version"]
                for r in await db.fetch_all("SELECT version FROM schema_migrations")
            }
            tables = {
                r["name"]
                for r in await db.fetch_all(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            await db.close()

        assert versions == {7}
        assert "thing" in tables
        assert "should_not_exist" not in tables


# ============================================================================
# Singleton lifecycle
# ============================================================================


class TestSingletonLifecycleUnit:
    @pytest.fixture(autouse=True)
    async def _clean_singleton(self):
        await close_database()
        yield
        await close_database()

    async def test_get_before_init_raises(self) -> None:
        with pytest.raises(RuntimeError):
            get_database()

    async def test_close_without_init_is_a_noop(self) -> None:
        await close_database()
        await close_database()

    async def test_init_is_idempotent(self, tmp_path) -> None:
        first = await init_db(str(tmp_path / "a.db"))
        second = await init_db(str(tmp_path / "ignored.db"))
        assert first is second

    async def test_close_then_init_opens_a_fresh_instance(self, tmp_path) -> None:
        first = await init_db(str(tmp_path / "a.db"))
        await close_database()

        with pytest.raises(RuntimeError):
            get_database()

        second = await init_db(str(tmp_path / "a.db"))
        assert second is not first

    async def test_init_uses_env_path_when_no_argument(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "env.db"
        monkeypatch.setenv("AGENTLENS_DB_PATH", str(target))

        db = await init_db()

        assert db._path == str(target)


# ============================================================================
# Backend selection
# ============================================================================


class TestBackendSelectionUnit:
    def test_defaults_to_memory(self, monkeypatch) -> None:
        monkeypatch.delenv("AGENTLENS_STORAGE", raising=False)
        assert web_api._backend() == "memory"

    def test_sqlite_value_is_case_insensitive(self, monkeypatch) -> None:
        monkeypatch.setenv("AGENTLENS_STORAGE", "SQLite")
        assert web_api._backend() == "sqlite"

    def test_unknown_value_is_passed_through(self, monkeypatch) -> None:
        monkeypatch.setenv("AGENTLENS_STORAGE", "postgres")
        assert web_api._backend() == "postgres"
