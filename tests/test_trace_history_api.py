"""
Tests for GET /api/traces (trace history / list endpoint).
"""
from __future__ import annotations
import base64
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def mock_agent_runner(monkeypatch, sample_trace, sample_event):
    from agentlens.web_api import _AgentRunner
    from uuid import uuid4

    async def fake_run(self, query):
        t_id = str(uuid4())
        e_id = str(uuid4())
        
        # Safely handle whether the teammate's fixture is a factory function or a direct object
        t_obj = sample_trace() if callable(sample_trace) else sample_trace
        e_obj = sample_event() if callable(sample_event) else sample_event
        
        # Create fresh linked copies (supports both Pydantic v1 and v2)
        if hasattr(t_obj, "model_copy"):
            t_obj = t_obj.model_copy(update={"trace_id": t_id, "event_count": 1})
            e_obj = e_obj.model_copy(update={"trace_id": t_id, "event_id": e_id})
        else:
            t_obj = t_obj.copy(update={"trace_id": t_id, "event_count": 1})
            e_obj = e_obj.copy(update={"trace_id": t_id, "event_id": e_id})
        
        # Grab live stores
        t_store = getattr(self, "trace_store", getattr(self, "tr_store", None))
        e_store = getattr(self, "event_store", getattr(self, "ev_store", None))
        
        if t_store:
            await t_store.save_trace(t_obj)
        if e_store:
            await e_store.save_event(e_obj)
            
        return {
            "trace_id": t_id,
            "agent_id": t_obj.agent_id,
            "status": "success",
            "event_count": 1,
            "total_latency_ms": 10.0,
            "response": "Offline Mock",
            "events": []
        }

    monkeypatch.setattr(_AgentRunner, "run", fake_run)

def _run(client: TestClient, query: str = "What is AGI?") -> dict[str, Any]:
    resp = client.post("/api/run", json={"query": query})
    assert resp.status_code == 200, resp.text
    return resp.json()

@pytest.fixture()
def sqlite_client(tmp_path, monkeypatch):
    db_file = tmp_path / "test_history.db"
    monkeypatch.setenv("AGENTLENS_STORAGE", "sqlite")
    monkeypatch.setenv("AGENTLENS_DB_PATH", str(db_file))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from agentlens.web_api import app
    with TestClient(app) as c:
        yield c

@pytest.fixture()
def memory_client(monkeypatch):
    monkeypatch.delenv("AGENTLENS_STORAGE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from agentlens.web_api import app, event_store, trace_store
    event_store._events.clear()
    event_store._trace_index.clear()
    trace_store._traces.clear()
    with TestClient(app) as c:
        yield c
    event_store._events.clear()
    event_store._trace_index.clear()
    trace_store._traces.clear()

class TestBasicListing:
    def test_empty_list(self, sqlite_client):
        assert sqlite_client.get("/api/traces").json()["count"] == 0
    def test_three_runs_returns_three(self, sqlite_client):
        for i in range(3): _run(sqlite_client, f"q{i}")
        body = sqlite_client.get("/api/traces").json()
        assert len(body["items"]) == 3
        assert body["count"] == 3
    def test_items_have_no_events_field(self, sqlite_client):
        _run(sqlite_client)
        assert "events" not in sqlite_client.get("/api/traces").json()["items"][0]
    def test_item_required_fields(self, sqlite_client):
        _run(sqlite_client)
        item = sqlite_client.get("/api/traces").json()["items"][0]
        for f in ("trace_id", "agent_id", "status", "start_time", "total_latency_ms", "event_count"):
            assert f in item
    def test_most_recent_first(self, sqlite_client):
        for i in range(3): _run(sqlite_client, f"q{i}")
        starts = [it["start_time"] for it in sqlite_client.get("/api/traces").json()["items"]]
        assert starts == sorted(starts, reverse=True)

class TestPagination:
    def test_limit_respected(self, sqlite_client):
        for i in range(5): _run(sqlite_client)
        body = sqlite_client.get("/api/traces?limit=2").json()
        assert len(body["items"]) == 2
        assert body["count"] == 5
    def test_cursor_fetches_next_page(self, sqlite_client):
        for i in range(3): _run(sqlite_client)
        page1 = sqlite_client.get("/api/traces?limit=2").json()
        page2 = sqlite_client.get(f"/api/traces?limit=2&cursor={page1['next_cursor']}").json()
        assert len(page2["items"]) == 1
    def test_no_duplicate_trace_ids(self, sqlite_client):
        for i in range(5): _run(sqlite_client)
        ids, cursor = [], None
        while True:
            body = sqlite_client.get("/api/traces?limit=2" + (f"&cursor={cursor}" if cursor else "")).json()
            ids.extend(it["trace_id"] for it in body["items"])
            if not (cursor := body.get("next_cursor")): break
        assert len(ids) == len(set(ids))
    def test_full_iteration_returns_all(self, sqlite_client):
        for i in range(5): _run(sqlite_client)
        ids, cursor = [], None
        while True:
            body = sqlite_client.get("/api/traces?limit=2" + (f"&cursor={cursor}" if cursor else "")).json()
            ids.extend(it["trace_id"] for it in body["items"])
            if not (cursor := body.get("next_cursor")): break
        assert len(ids) == 5

class TestFilters:
    def test_status_all_equals_no_filter(self, sqlite_client):
        for i in range(2): _run(sqlite_client)
        assert sqlite_client.get("/api/traces").json()["count"] == sqlite_client.get("/api/traces?status=all").json()["count"]
    def test_status_success_filter(self, sqlite_client):
        for i in range(3): _run(sqlite_client)
        assert sqlite_client.get("/api/traces?status=success").json()["count"] == 3
    def test_status_failure_filter_returns_zero(self, sqlite_client):
        for i in range(2): _run(sqlite_client)
        assert sqlite_client.get("/api/traces?status=failure").json()["count"] == 0
    def test_agent_id_filter_known(self, sqlite_client):
        for i in range(2): _run(sqlite_client)
        assert sqlite_client.get("/api/traces?agent_id=multistep-research-agent").json()["count"] == 2
    def test_agent_id_filter_unknown(self, sqlite_client):
        _run(sqlite_client)
        assert sqlite_client.get("/api/traces?agent_id=ghost").json()["count"] == 0
    def test_start_date_today(self, sqlite_client):
        _run(sqlite_client)
        assert sqlite_client.get(f"/api/traces?start_date={datetime.now(timezone.utc).strftime('%Y-%m-%d')}").json()["count"] == 1
    def test_start_date_future_returns_empty(self, sqlite_client):
        _run(sqlite_client)
        assert sqlite_client.get("/api/traces?start_date=2099-12-31").json()["count"] == 0
    def test_end_date_today(self, sqlite_client):
        _run(sqlite_client)
        assert sqlite_client.get(f"/api/traces?end_date={datetime.now(timezone.utc).strftime('%Y-%m-%d')}").json()["count"] == 1
    def test_end_date_past_returns_empty(self, sqlite_client):
        _run(sqlite_client)
        assert sqlite_client.get("/api/traces?end_date=2000-01-01").json()["count"] == 0
    def test_start_date_iso8601(self, sqlite_client):
        _run(sqlite_client)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")
        assert sqlite_client.get(f"/api/traces?start_date={now_iso}").json()["count"] == 1
    def test_limit_param(self, sqlite_client):
        for i in range(5): _run(sqlite_client)
        assert len(sqlite_client.get("/api/traces?limit=3").json()["items"]) == 3

class TestErrors:
    def test_invalid_status_returns_400(self, sqlite_client):
        assert sqlite_client.get("/api/traces?status=broken").status_code == 400
    def test_valid_statuses_return_200(self, sqlite_client):
        for s in ("all", "success", "failure", "partial", "pending"):
            assert sqlite_client.get(f"/api/traces?status={s}").status_code == 200
    def test_invalid_start_date_returns_400(self, sqlite_client):
        assert sqlite_client.get("/api/traces?start_date=not-a-date").status_code == 400
    def test_invalid_end_date_returns_400(self, sqlite_client):
        assert sqlite_client.get("/api/traces?end_date=20231301").status_code == 400
    def test_malformed_cursor_returns_400(self, sqlite_client):
        assert sqlite_client.get("/api/traces?cursor=!!!not_base64!!!").status_code == 400
    def test_cursor_without_separator_returns_400(self, sqlite_client):
        assert sqlite_client.get(f"/api/traces?cursor={base64.urlsafe_b64encode(b'nodatepipe').decode()}").status_code == 400
    def test_limit_too_large_returns_422(self, sqlite_client):
        assert sqlite_client.get("/api/traces?limit=999").status_code == 422
    def test_limit_zero_returns_422(self, sqlite_client):
        assert sqlite_client.get("/api/traces?limit=0").status_code == 422

class TestGetTraceById:
    def test_full_events_returned_sqlite(self, sqlite_client):
        run = _run(sqlite_client)
        resp = sqlite_client.get(f"/api/traces/{run['trace_id']}")
        assert resp.status_code == 200
        assert len(resp.json()["events"]) > 0
    def test_not_found_returns_404(self, sqlite_client):
        assert sqlite_client.get(f"/api/traces/{uuid4()}").status_code == 404
    def test_invalid_trace_id_returns_400(self, sqlite_client):
        assert sqlite_client.get("/api/traces/not-a-uuid").status_code == 400

class TestMemoryBackend:
    def test_run_and_list(self, memory_client):
        run = _run(memory_client)
        assert memory_client.get("/api/traces").json()["count"] == 1
    def test_pagination(self, memory_client):
        for i in range(4): _run(memory_client)
        assert len(memory_client.get("/api/traces?limit=2").json()["items"]) == 2
    def test_invalid_status_400(self, memory_client):
        assert memory_client.get("/api/traces?status=banana").status_code == 400
