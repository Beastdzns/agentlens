"""
AgentLens Web API
=================

FastAPI application that exposes the multi-step research agent over HTTP.

Endpoints
---------
- ``POST /api/run``              — Run the agent with a query, return the full trace.
- ``GET  /api/traces/{trace_id}`` — Retrieve a previously stored trace by ID.

Configuration
-------------
- ``GEMINI_API_KEY`` environment variable — required for live Gemini calls.
- ``CORS_ORIGINS``   environment variable — comma-separated allowed origins
  (defaults to ``*`` for local development).

Start the server::

    uvicorn agentlens.web_api:app --reload
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agentlens.events import EventStatus, EventType, Trace
from agentlens.sdk_collector import EventCollector
from agentlens.storage import InMemoryEventStore, InMemoryTraceStore

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgentLens API",
    description="HTTP interface for the AgentLens multi-step research agent.",
    version="0.1.0",
)

# CORS — read allowed origins from env, default to permissive for dev.
_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static frontend assets (if frontend directory exists)
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

# ---------------------------------------------------------------------------
# Shared in-memory stores (module-level singletons for the lifetime of the
# server process).  A production deployment would swap these for persistent
# backends.
# ---------------------------------------------------------------------------

event_store = InMemoryEventStore()
trace_store = InMemoryTraceStore()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Body for ``POST /api/run``."""

    query: str = Field(..., min_length=1, description="The research query to run.")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class SerializedEvent(BaseModel):
    """Wire-format for a single agent event."""

    event_id: str
    event_type: str
    status: str
    input_data: Optional[dict[str, Any]] = None
    output_data: Optional[dict[str, Any]] = None
    latency_ms: Optional[float] = None
    parent_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class RunResponse(BaseModel):
    """Body returned by ``POST /api/run``."""

    trace_id: str
    response: str
    events: list[SerializedEvent]
    total_latency_ms: float
    status: str


class TraceResponse(BaseModel):
    """Body returned by ``GET /api/traces/{trace_id}``."""

    trace_id: str
    agent_id: str
    status: str
    event_count: int
    total_latency_ms: float
    events: list[SerializedEvent]


# ---------------------------------------------------------------------------
# Agent runner (encapsulates the multi-step workflow)
# ---------------------------------------------------------------------------


class _AgentRunner:
    """Runs the multi-step research agent and returns a completed Trace."""

    def __init__(
        self,
        ev_store: InMemoryEventStore,
        tr_store: InMemoryTraceStore,
    ) -> None:
        self._event_store = ev_store
        self._trace_store = tr_store

    async def run(self, query: str) -> Trace:
        """Execute the full pipeline and return a finalised Trace."""
        collector = EventCollector(storage_backend=self._event_store)

        # --- Initialise Gemini (if key is available) -----------------------
        gemini_model: Any = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai

                gemini_model = genai.Client(api_key=api_key)
            except ImportError:
                pass  # dependency not installed — use fallback

        async def _ask_gemini(prompt: str, fallback: str) -> str:
            if gemini_model is None:
                return fallback
            try:
                resp = await asyncio.to_thread(
                    gemini_model.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                return str(resp.text)
            except Exception:
                raise  # let caller handle Gemini failures

        # --- Start trace ---------------------------------------------------
        trace_id = await collector.start_trace(agent_id="multistep-research-agent")
        started = time.perf_counter()
        answer = ""

        try:
            # Step 1 — Planning
            planning_prompt = f"Create a concise three-step research plan for: {query}"
            plan = await _ask_gemini(
                planning_prompt,
                "1. Search current sources 2. Retrieve useful evidence 3. Synthesize findings",
            )
            planning_event = await collector.record_event(
                event_type=EventType.DECISION,
                input_data={"query": query},
                output_data={"plan": plan},
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata={"step": "planning"},
            )

            # Step 2 — Web search (simulated tool call)
            collector.push_parent(planning_event.event_id)
            search_started = time.perf_counter()
            search_event = await collector.record_event(
                event_type=EventType.TOOL_CALL,
                input_data={"tool": "web_search", "query": query},
                output_data={
                    "hits": [
                        "research-paper-1",
                        "research-paper-2",
                        "technical-report-1",
                    ]
                },
                latency_ms=(time.perf_counter() - search_started) * 1000,
                metadata={"step": "search", "tool": "web_search"},
            )

            # Step 3 — Retrieval
            collector.push_parent(search_event.event_id)
            retrieval_started = time.perf_counter()
            retrieval_event = await collector.record_event(
                event_type=EventType.RETRIEVAL,
                input_data={"top_k": 3, "query": query},
                output_data={
                    "documents": [
                        {"id": "research-paper-1", "relevance": 0.94},
                        {"id": "research-paper-2", "relevance": 0.87},
                        {"id": "technical-report-1", "relevance": 0.81},
                    ]
                },
                latency_ms=(time.perf_counter() - retrieval_started) * 1000,
                metadata={"step": "retrieval"},
            )
            collector.pop_parent()

            # Step 4 — Memory write
            await collector.record_event(
                event_type=EventType.MEMORY_WRITE,
                input_data={"key": "retrieved_evidence"},
                output_data={"document_count": len(retrieval_event.output_data or {})},
                latency_ms=2.0,
                metadata={"step": "memory-write"},
            )
            collector.pop_parent()

            # Step 5 — Synthesis LLM call
            synthesis_started = time.perf_counter()
            synthesis_prompt = (
                f"Answer this query using the plan and evidence below:\n"
                f"Query: {query}\nPlan: {plan}\n"
                f"Evidence: {retrieval_event.output_data}"
            )
            answer = await _ask_gemini(
                synthesis_prompt,
                "The evidence suggests a measurable, traceable workflow with "
                "planning, retrieval, and synthesis.",
            )
            await collector.record_event(
                event_type=EventType.LLM_CALL,
                input_data={"prompt": query, "plan": plan},
                output_data={"response": answer},
                latency_ms=(time.perf_counter() - synthesis_started) * 1000,
                metadata={"step": "synthesis", "model": "gemini-2.5-flash"},
            )

            # Step 6 — Final response
            await collector.record_event(
                event_type=EventType.FINAL_RESPONSE,
                output_data={"answer": answer},
                latency_ms=0.0,
                metadata={"step": "final-response"},
            )

            await collector.end_trace()

        except Exception as exc:
            await collector.record_event(
                event_type=EventType.ERROR,
                status=EventStatus.FAILURE,
                output_data={"error": str(exc)},
                metadata={"exception_type": type(exc).__name__},
            )
            await collector.end_trace()

            failed_events = await self._event_store.get_trace_events(trace_id)
            failed_trace = Trace(
                trace_id=trace_id,
                agent_id="multistep-research-agent",
                status=EventStatus.FAILURE,
                events=failed_events,
                event_count=len(failed_events),
                total_latency_ms=sum(e.latency_ms or 0.0 for e in failed_events),
            )
            failed_trace.finalize(EventStatus.FAILURE)
            await self._trace_store.save_trace(failed_trace)
            raise

        # --- Assemble and persist the Trace object -------------------------
        stored_events = await self._event_store.get_trace_events(trace_id)
        trace = Trace(
            trace_id=trace_id,
            agent_id="multistep-research-agent",
            status=EventStatus.SUCCESS,
            events=stored_events,
            event_count=len(stored_events),
            total_latency_ms=sum(e.latency_ms or 0.0 for e in stored_events),
        )
        trace.finalize(EventStatus.SUCCESS)
        await self._trace_store.save_trace(trace)
        return trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_event(event: Any) -> SerializedEvent:
    """Convert an AgentEvent to the wire-format dict."""
    ts = None
    if hasattr(event, "timestamp") and event.timestamp is not None:
        ts = (
            event.timestamp.isoformat()
            if hasattr(event.timestamp, "isoformat")
            else str(event.timestamp)
        )

    return SerializedEvent(
        event_id=str(event.event_id),
        event_type=(
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        ),
        status=event.status.value if hasattr(event.status, "value") else str(event.status),
        input_data=event.input_data,
        output_data=event.output_data,
        latency_ms=event.latency_ms,
        parent_id=str(event.parent_id) if event.parent_id else None,
        agent_id=getattr(event, "agent_id", None),
        timestamp=ts,
        metadata=getattr(event, "metadata", None),
    )


def _extract_answer(trace: Trace) -> str:
    """Pull the final answer text from the trace events."""
    for event in reversed(trace.events):
        if (
            event.event_type == EventType.FINAL_RESPONSE
            and event.output_data
            and "answer" in event.output_data
        ):
            return str(event.output_data["answer"])
    # Fallback: look for the last LLM_CALL response.
    for event in reversed(trace.events):
        if (
            event.event_type == EventType.LLM_CALL
            and event.output_data
            and "response" in event.output_data
        ):
            return str(event.output_data["response"])
    return ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint polled by the frontend dashboard."""
    return {"status": "ok", "app": "AgentLens API", "version": "0.1.0"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root info endpoint providing service status and navigation links."""
    return {
        "app": "AgentLens API",
        "status": "online",
        "docs": "/docs",
        "dashboard": "/app/",
    }


@app.post("/api/run", response_model=RunResponse)
async def run_agent(body: RunRequest) -> RunResponse:
    """Run the multi-step research agent and return the complete trace."""
    runner = _AgentRunner(event_store, trace_store)

    try:
        trace = await runner.run(body.query)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agent execution failed: {exc}",
        ) from exc

    return RunResponse(
        trace_id=str(trace.trace_id),
        response=_extract_answer(trace),
        events=[_serialize_event(e) for e in trace.events],
        total_latency_ms=trace.total_latency_ms,
        status="success",
    )


@app.get("/api/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str) -> TraceResponse:
    """Retrieve a previously stored trace by its UUID."""
    try:
        tid = UUID(trace_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid trace_id format: {trace_id}",
        ) from exc

    try:
        trace = await trace_store.get_trace(tid)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Trace {trace_id} not found",
        ) from exc

    return TraceResponse(
        trace_id=str(trace.trace_id),
        agent_id=trace.agent_id,
        status=trace.status.value if hasattr(trace.status, "value") else str(trace.status),
        event_count=trace.event_count,
        total_latency_ms=trace.total_latency_ms,
        events=[_serialize_event(e) for e in trace.events],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agentlens.web_api:app", host="0.0.0.0", port=8000, reload=True)
