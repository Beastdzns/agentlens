"""
Tests for ``GET /api/traces`` (trace history / list endpoint).

These exercise the *real* ``_AgentRunner`` pipeline.  With ``GEMINI_API_KEY``
unset the runner takes its deterministic offline fallback, so no network is
required and every run succeeds and is persisted to the configured store.

The suite runs the endpoint against **both** the SQLite backend (persistent)
and the default in-memory backend (regression), so the two stay in parity.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from agentlens.web_api import app


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _isolate(monkeypatch) -> Iterator[None]:
    """Force deterministic offline runs and start from fresh in-memory stores.

    The shared default singletons are *cleared*, never rebound: the lifespan
    restores these exact objects after a SQLite session, and other test
    modules import them by reference.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTLENS_STORAGE", raising=False)

    import agentlens.web_api as web_api

    def _clear() -> None:
        web_api.event_store._events.clear()
        web_api.event_store._trace_index.clear()
        web_api.trace_store._traces.clear()

    _clear()
    yield
    _clear()


@pytest.fixture()
def sqlite_client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    """Test client backed by a fresh, isolated SQLite database."""
    monkeypatch.setenv("AGENTLENS_STORAGE", "sqlite")
    monkeypatch.setenv("AGENTLENS_DB_PATH", str(tmp_path / "test_history.db"))
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def memory_client(monkeypatch) -> Iterator[TestClient]:
    """Test client using the default in-memory backend (regression)."""
    monkeypatch.delenv("AGENTLENS_STORAGE", raising=False)
    with TestClient(app) as client:
        yield client


# ===========================================================================
# Helpers
# ===========================================================================


def _run(client: TestClient, query: str = "What is AGI?") -> dict[str, Any]:
    """Execute an offline agent run and return the parsed ``/api/run`` body."""
    resp = client.post("/api/run", json={"query": query})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _list(client: TestClient, **params: Any) -> dict[str, Any]:
    resp = client.get("/api/traces", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _collect_all_pages(client: TestClient, limit: int = 2) -> list[str]:
    """Follow keyset cursors to exhaustion and return every trace_id seen."""
    ids: list[str] = []
    cursor: Optional[str] = None
    while True:
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        body = _list(client, **params)
        ids.extend(item["trace_id"] for item in body["items"])
        cursor = body.get("next_cursor")
        if not cursor:
            return ids


# ===========================================================================
# Basic listing
# ===========================================================================


class TestBasicListing:
    def test_empty_list(self, sqlite_client: TestClient) -> None:
        body = _list(sqlite_client)
        assert body["items"] == []
        assert body["count"] == 0
        assert body["next_cursor"] is None

    def test_three_runs_returns_three(self, sqlite_client: TestClient) -> None:
        for i in range(3):
            _run(sqlite_client, f"q{i}")
        body = _list(sqlite_client)
        assert len(body["items"]) == 3
        assert body["count"] == 3

    def test_items_are_summaries_without_events(self, sqlite_client: TestClient) -> None:
        _run(sqlite_client)
        item = _list(sqlite_client)["items"][0]
        assert "events" not in item

    def test_item_required_fields(self, sqlite_client: TestClient) -> None:
        _run(sqlite_client)
        item = _list(sqlite_client)["items"][0]
        for field in (
            "trace_id",
            "agent_id",
            "status",
            "start_time",
            "end_time",
            "total_latency_ms",
            "event_count",
        ):
            assert field in item, field

    def test_most_recent_first(self, sqlite_client: TestClient) -> None:
        for i in range(3):
            _run(sqlite_client, f"q{i}")
        starts = [item["start_time"] for item in _list(sqlite_client)["items"]]
        assert starts == sorted(starts, reverse=True)


# ===========================================================================
# Pagination
# ===========================================================================


class TestPagination:
    def test_limit_is_respected(self, sqlite_client: TestClient) -> None:
        for _ in range(5):
            _run(sqlite_client)
        body = _list(sqlite_client, limit=2)
        assert len(body["items"]) == 2
        assert body["count"] == 5
        assert body["next_cursor"] is not None

    def test_cursor_fetches_next_page(self, sqlite_client: TestClient) -> None:
        for _ in range(3):
            _run(sqlite_client)
        page1 = _list(sqlite_client, limit=2)
        page2 = _list(sqlite_client, limit=2, cursor=page1["next_cursor"])
        assert len(page2["items"]) == 1
        assert page2["next_cursor"] is None

    def test_no_duplicate_or_missing_trace_ids(self, sqlite_client: TestClient) -> None:
        expected = {_run(sqlite_client, f"q{i}")["trace_id"] for i in range(5)}
        seen = _collect_all_pages(sqlite_client, limit=2)
        assert len(seen) == len(set(seen)) == 5
        assert set(seen) == expected


# ===========================================================================
# Filters
# ===========================================================================


class TestFilters:
    def test_status_all_equals_no_filter(self, sqlite_client: TestClient) -> None:
        for _ in range(2):
            _run(sqlite_client)
        assert _list(sqlite_client)["count"] == _list(sqlite_client, status="all")["count"]

    def test_status_success_filter(self, sqlite_client: TestClient) -> None:
        # Offline runs always complete successfully.
        for _ in range(3):
            _run(sqlite_client)
        assert _list(sqlite_client, status="success")["count"] == 3

    def test_status_failure_filter_returns_zero(self, sqlite_client: TestClient) -> None:
        for _ in range(2):
            _run(sqlite_client)
        assert _list(sqlite_client, status="failure")["count"] == 0

    def test_agent_id_filter_known(self, sqlite_client: TestClient) -> None:
        for _ in range(2):
            _run(sqlite_client)
        assert _list(sqlite_client, agent_id="multistep-research-agent")["count"] == 2

    def test_agent_id_filter_unknown(self, sqlite_client: TestClient) -> None:
        _run(sqlite_client)
        assert _list(sqlite_client, agent_id="ghost")["count"] == 0

    def test_start_date_today_includes_run(self, sqlite_client: TestClient) -> None:
        _run(sqlite_client)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert _list(sqlite_client, start_date=today)["count"] == 1

    def test_start_date_in_future_excludes_run(self, sqlite_client: TestClient) -> None:
        _run(sqlite_client)
        assert _list(sqlite_client, start_date="2099-12-31")["count"] == 0

    def test_end_date_today_includes_run(self, sqlite_client: TestClient) -> None:
        _run(sqlite_client)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert _list(sqlite_client, end_date=today)["count"] == 1

    def test_end_date_in_past_excludes_run(self, sqlite_client: TestClient) -> None:
        _run(sqlite_client)
        assert _list(sqlite_client, end_date="2000-01-01")["count"] == 0

    def test_start_date_accepts_iso8601(self, sqlite_client: TestClient) -> None:
        _run(sqlite_client)
        midnight = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")
        assert _list(sqlite_client, start_date=midnight)["count"] == 1

    def test_limit_param(self, sqlite_client: TestClient) -> None:
        for _ in range(5):
            _run(sqlite_client)
        assert len(_list(sqlite_client, limit=3)["items"]) == 3


# ===========================================================================
# Validation / errors
# ===========================================================================


class TestErrors:
    def test_invalid_status_returns_400(self, sqlite_client: TestClient) -> None:
        assert sqlite_client.get("/api/traces", params={"status": "broken"}).status_code == 400

    def test_valid_statuses_return_200(self, sqlite_client: TestClient) -> None:
        for status in ("all", "success", "failure", "partial", "pending"):
            assert sqlite_client.get("/api/traces", params={"status": status}).status_code == 200

    def test_invalid_start_date_returns_400(self, sqlite_client: TestClient) -> None:
        assert sqlite_client.get("/api/traces", params={"start_date": "not-a-date"}).status_code == 400

    def test_invalid_end_date_returns_400(self, sqlite_client: TestClient) -> None:
        assert sqlite_client.get("/api/traces", params={"end_date": "20231301"}).status_code == 400

    def test_malformed_cursor_returns_400(self, sqlite_client: TestClient) -> None:
        assert sqlite_client.get("/api/traces", params={"cursor": "!!!not_base64!!!"}).status_code == 400

    def test_cursor_without_separator_returns_400(self, sqlite_client: TestClient) -> None:
        bad = base64.urlsafe_b64encode(b"nodatepipe").decode()
        assert sqlite_client.get("/api/traces", params={"cursor": bad}).status_code == 400

    def test_limit_too_large_returns_422(self, sqlite_client: TestClient) -> None:
        assert sqlite_client.get("/api/traces", params={"limit": 999}).status_code == 422

    def test_limit_zero_returns_422(self, sqlite_client: TestClient) -> None:
        assert sqlite_client.get("/api/traces", params={"limit": 0}).status_code == 422


# ===========================================================================
# Single-trace endpoint (regression: still returns full events)
# ===========================================================================


class TestGetTraceById:
    def test_full_events_returned(self, sqlite_client: TestClient) -> None:
        run = _run(sqlite_client)
        resp = sqlite_client.get(f"/api/traces/{run['trace_id']}")
        assert resp.status_code == 200
        assert len(resp.json()["events"]) > 0

    def test_not_found_returns_404(self, sqlite_client: TestClient) -> None:
        from uuid import uuid4

        assert sqlite_client.get(f"/api/traces/{uuid4()}").status_code == 404

    def test_invalid_trace_id_returns_400(self, sqlite_client: TestClient) -> None:
        assert sqlite_client.get("/api/traces/not-a-uuid").status_code == 400


# ===========================================================================
# Persistence across a server restart (lifespan start/stop in one process)
# ===========================================================================


class TestPersistence:
    def test_traces_survive_restart(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("AGENTLENS_STORAGE", "sqlite")
        monkeypatch.setenv("AGENTLENS_DB_PATH", str(tmp_path / "persist.db"))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with TestClient(app) as client:
            trace_id = _run(client)["trace_id"]

        # A brand-new lifespan must reopen the *same* database file.
        with TestClient(app) as client:
            body = _list(client)
            assert body["count"] == 1
            assert body["items"][0]["trace_id"] == trace_id


# ===========================================================================
# Default (in-memory) backend regression
# ===========================================================================


class TestMemoryBackend:
    def test_run_and_list(self, memory_client: TestClient) -> None:
        _run(memory_client)
        assert _list(memory_client)["count"] == 1

    def test_pagination(self, memory_client: TestClient) -> None:
        for _ in range(4):
            _run(memory_client)
        body = _list(memory_client, limit=2)
        assert len(body["items"]) == 2
        assert body["next_cursor"] is not None

    def test_invalid_status_400(self, memory_client: TestClient) -> None:
        assert memory_client.get("/api/traces", params={"status": "banana"}).status_code == 400

    def test_full_iteration(self, memory_client: TestClient) -> None:
        expected = {_run(memory_client, f"q{i}")["trace_id"] for i in range(4)}
        seen = _collect_all_pages(memory_client, limit=2)
        assert set(seen) == expected
