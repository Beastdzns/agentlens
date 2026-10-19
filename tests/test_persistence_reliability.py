"""
End-to-end persistence reliability for the SQLite backend (Phase 9, Member 3).

The store-level tests live in ``test_storage_sqlite.py``.  These tests drive
the **whole application** through repeated start/stop cycles (the FastAPI
lifespan) to prove the properties a real deployment depends on:

- traces *and* their events survive a server restart;
- repeated start/stop cycles reuse the same database file (regression for the
  singleton never being reset, which used to deadlock the read pool);
- migrations run on startup;
- the default in-memory backend is untouched by any of this.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

import agentlens.web_api as web_api
from agentlens.storage.db import close_database, get_database, init_db
from agentlens.web_api import app

pytestmark = pytest.mark.integration


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def sqlite_app(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite file and force offline runs."""
    db_file = tmp_path / "reliability.db"
    monkeypatch.setenv("AGENTLENS_STORAGE", "sqlite")
    monkeypatch.setenv("AGENTLENS_DB_PATH", str(db_file))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return db_file


def _run(client: TestClient, query: str = "reliability query") -> dict:
    resp = client.post("/api/run", json={"query": query})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ===========================================================================
# Restart recovery
# ===========================================================================


class TestRestartRecovery:
    def test_trace_and_events_survive_restart(self, sqlite_app) -> None:
        with TestClient(app) as client:
            run = _run(client)
            trace_id = run["trace_id"]
            before = client.get(f"/api/traces/{trace_id}").json()["events"]
            assert len(before) > 0

        # Brand-new lifespan (simulates a server restart).
        with TestClient(app) as client:
            detail = client.get(f"/api/traces/{trace_id}")
            assert detail.status_code == 200, detail.text
            after = detail.json()["events"]
            assert [e["event_id"] for e in after] == [e["event_id"] for e in before]
            assert client.get("/api/traces").json()["count"] == 1

    def test_repeated_start_stop_cycles_accumulate(self, sqlite_app) -> None:
        # Regression: the old singleton bug made the 2nd cycle reuse a closed
        # database (read-pool deadlock).  Every cycle must open cleanly.
        with TestClient(app) as client:
            _run(client, "cycle-1")
        with TestClient(app) as client:
            assert client.get("/api/traces").json()["count"] == 1
            _run(client, "cycle-2")
        with TestClient(app) as client:
            assert client.get("/api/traces").json()["count"] == 2
        with TestClient(app) as client:
            assert client.get("/api/traces").json()["count"] == 2


# ===========================================================================
# Startup schema
# ===========================================================================


class TestStartupSchema:
    def test_migrations_run_on_startup(self, sqlite_app) -> None:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

        con = sqlite3.connect(str(sqlite_app))
        try:
            versions = sorted(r[0] for r in con.execute("SELECT version FROM schema_migrations"))
            tables = {
                r[0]
                for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        finally:
            con.close()

        assert versions == [1]
        assert {"events", "traces", "schema_migrations"} <= tables

    def test_data_file_is_reused_not_recreated(self, sqlite_app) -> None:
        with TestClient(app) as client:
            _run(client)
        size_after_first = sqlite_app.stat().st_size

        with TestClient(app) as client:
            assert client.get("/api/traces").json()["count"] == 1

        # Same file, no data loss, no "fresh" database.
        assert sqlite_app.stat().st_size >= size_after_first


# ===========================================================================
# Database singleton lifecycle
# ===========================================================================


class TestSingletonLifecycle:
    async def test_close_database_resets_singleton(self, tmp_path) -> None:
        await close_database()

        db1 = await init_db(str(tmp_path / "life.db"))
        assert get_database() is db1

        await close_database()
        with pytest.raises(RuntimeError):
            get_database()

        # A new init_db opens a *fresh* instance rather than the closed one.
        db2 = await init_db(str(tmp_path / "life.db"))
        assert db2 is not db1
        await close_database()

    async def test_close_database_is_idempotent(self) -> None:
        await close_database()
        await close_database()  # must not raise


# ===========================================================================
# Backend isolation (default stays in-memory)
# ===========================================================================


class TestBackendIsolation:
    def test_default_backend_is_memory_and_persists_nothing(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("AGENTLENS_STORAGE", raising=False)
        monkeypatch.setenv("AGENTLENS_DB_PATH", str(tmp_path / "unused.db"))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # The stores used by the app must be the in-memory ones.
        assert web_api.event_store.__class__.__name__ == "InMemoryEventStore"

        with TestClient(app) as client:
            _run(client)
            assert client.get("/api/traces").json()["count"] == 1

        # No SQLite file should have been created for a memory-backed run.
        assert not (tmp_path / "unused.db").exists()
