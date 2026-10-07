"""
Tests for the AgentLens Web API (``agentlens.web_api``).

Covers:
- Successful agent run (``POST /api/run``)
- Empty / blank query validation
- Gemini failure propagation (502)
- Trace retrieval (``GET /api/traces/{trace_id}``)
- Trace not found (404)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agentlens.web_api import app, event_store, trace_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_stores():
    """Reset the module-level stores before every test."""
    event_store._events.clear()
    event_store._trace_index.clear()
    trace_store._traces.clear()
    yield
    event_store._events.clear()
    event_store._trace_index.clear()
    trace_store._traces.clear()


@pytest.fixture
def client():
    """Synchronous test client for FastAPI."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/run — successful run (offline / no Gemini key)
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    """Tests for POST /api/run."""

    def test_successful_run(self, client: TestClient):
        """A valid query should return a complete trace with all event fields."""
        response = client.post(
            "/api/run",
            json={"query": "How can AI agents be debugged?"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["trace_id"]  # non-empty UUID string
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        assert isinstance(data["events"], list)
        assert len(data["events"]) > 0
        assert isinstance(data["total_latency_ms"], (int, float))
        assert data["total_latency_ms"] >= 0

        # Verify each event has the required serialized fields.
        for event in data["events"]:
            assert "event_id" in event
            assert "event_type" in event
            assert "status" in event
            # input_data / output_data may be None for certain event types
            assert "latency_ms" in event
            assert "parent_id" in event  # may be null

    def test_successful_run_stores_trace(self, client: TestClient):
        """After POST /api/run the trace should be retrievable via GET."""
        run_resp = client.post(
            "/api/run",
            json={"query": "What is observability?"},
        )
        assert run_resp.status_code == 200
        trace_id = run_resp.json()["trace_id"]

        get_resp = client.get(f"/api/traces/{trace_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["trace_id"] == trace_id

    def test_empty_query_returns_422(self, client: TestClient):
        """An empty string query should be rejected by Pydantic validation."""
        response = client.post("/api/run", json={"query": ""})
        assert response.status_code == 422

    def test_blank_query_returns_422(self, client: TestClient):
        """A whitespace-only query should be rejected."""
        response = client.post("/api/run", json={"query": "   "})
        assert response.status_code == 422

    def test_missing_query_returns_422(self, client: TestClient):
        """Omitting the query field entirely should be rejected."""
        response = client.post("/api/run", json={})
        assert response.status_code == 422

    def test_gemini_failure_returns_502(self, client: TestClient):
        """When Gemini raises an exception the API should return 502."""
        # Patch os.getenv so the runner thinks a key is available, then make
        # the Gemini model raise.
        fake_model = MagicMock()
        fake_model.generate_content.side_effect = RuntimeError("Gemini quota exceeded")

        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            with patch("agentlens.web_api.os.getenv", side_effect=lambda k, *a: {
                "GEMINI_API_KEY": "fake-key",
                "CORS_ORIGINS": "*",
            }.get(k, a[0] if a else None)):
                with patch(
                    "google.generativeai.GenerativeModel",
                    return_value=fake_model,
                ):
                    with patch("google.generativeai.configure"):
                        response = client.post(
                            "/api/run",
                            json={"query": "trigger gemini failure"},
                        )

        assert response.status_code == 502
        assert "Agent execution failed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/traces/{trace_id}
# ---------------------------------------------------------------------------


class TestTraceEndpoint:
    """Tests for GET /api/traces/{trace_id}."""

    def test_trace_retrieval_after_run(self, client: TestClient):
        """Running the agent then fetching the trace should succeed."""
        run_resp = client.post(
            "/api/run",
            json={"query": "Explain causal tracing."},
        )
        assert run_resp.status_code == 200
        trace_id = run_resp.json()["trace_id"]

        trace_resp = client.get(f"/api/traces/{trace_id}")
        assert trace_resp.status_code == 200

        data = trace_resp.json()
        assert data["trace_id"] == trace_id
        assert data["agent_id"] == "multistep-research-agent"
        assert data["status"] == "success"
        assert data["event_count"] > 0
        assert data["total_latency_ms"] >= 0
        assert isinstance(data["events"], list)
        assert len(data["events"]) == data["event_count"]

    def test_trace_not_found_returns_404(self, client: TestClient):
        """Requesting a non-existent trace should return 404."""
        fake_id = str(uuid4())
        response = client.get(f"/api/traces/{fake_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_invalid_trace_id_returns_400(self, client: TestClient):
        """A malformed UUID should return 400."""
        response = client.get("/api/traces/not-a-uuid")
        assert response.status_code == 400
        assert "Invalid trace_id" in response.json()["detail"]
