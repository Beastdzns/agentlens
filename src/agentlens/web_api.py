"""
AgentLens Web API
=================

FastAPI application that exposes the multi-step research agent over HTTP.

Endpoints
---------
- ``POST /api/run``              — Run the agent with a query, return the full trace.
- ``GET  /api/traces/{trace_id}`` — Retrieve a previously stored trace by ID.
  Optional query params:
    - ``status``     — Filter events by status (all | success | failure | partial | pending).
    - ``event_type`` — Filter events by type (llm_call | tool_call | retrieval | …).

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
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agentlens.events import EventStatus, EventType, Trace
from agentlens.sdk_collector import ContextManager, EventCollector, EventNormalizer
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

# ---------------------------------------------------------------------------
# Shared in-memory stores (module-level singletons for the lifetime of the
# server process).  A production deployment would swap these for persistent
# backends.
# ---------------------------------------------------------------------------

event_store = InMemoryEventStore()
trace_store = InMemoryTraceStore()

# ---------------------------------------------------------------------------
# Valid filter value sets
# ---------------------------------------------------------------------------

_VALID_STATUS_FILTERS: frozenset[str] = frozenset(
    {"all", "success", "failure", "partial", "pending"}
)

_VALID_EVENT_TYPES: frozenset[str] = frozenset(
    et.value for et in EventType
)


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
    """Wire-format for a single agent event — complete event data."""

    event_id: str
    trace_id: str
    parent_id: Optional[str] = None
    agent_id: str
    event_type: str
    status: str
    timestamp: str
    input_data: Optional[dict[str, Any]] = None
    output_data: Optional[dict[str, Any]] = None
    latency_ms: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    # Derived convenience field — list of direct children event IDs.
    children: list[str] = Field(default_factory=list)


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
    # Summary counts
    success_count: int = 0
    failure_count: int = 0
    partial_count: int = 0
    pending_count: int = 0
    failure_event_ids: list[str] = Field(default_factory=list)
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
        context = ContextManager()

        # --- Initialise Gemini (if key is available) -----------------------
        gemini_model: Any = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            except ImportError:
                pass  # dependency not installed — use fallback

        async def _ask_gemini(prompt: str, fallback: str) -> str:
            if gemini_model is None:
                return fallback
            try:
                resp = await asyncio.to_thread(
                    gemini_model.generate_content, prompt,
                )
                return resp.text
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
            context.push_parent(planning_event.event_id)
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
            context.push_parent(search_event.event_id)
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
            context.pop_parent()

            # Step 4 — Memory write
            await collector.record_event(
                event_type=EventType.MEMORY_WRITE,
                input_data={"key": "retrieved_evidence"},
                output_data={
                    "document_count": len(retrieval_event.output_data or {})
                },
                latency_ms=2.0,
                metadata={"step": "memory-write"},
            )
            context.pop_parent()

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

        except Exception:
            # End the trace so context is cleaned up, then re-raise.
            await collector.end_trace()
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


def _build_children_map(events: list[Any]) -> dict[str, list[str]]:
    """Return a mapping of event_id -> list of child event_ids."""
    children: dict[str, list[str]] = {}
    for event in events:
        eid = str(event.event_id)
        children.setdefault(eid, [])
        if event.parent_id:
            pid = str(event.parent_id)
            children.setdefault(pid, [])
            children[pid].append(eid)
    return children


def _serialize_event(event: Any, children_map: dict[str, list[str]] | None = None) -> SerializedEvent:
    """Convert an AgentEvent to the complete wire-format dict."""
    eid = str(event.event_id)
    return SerializedEvent(
        event_id=eid,
        trace_id=str(event.trace_id),
        parent_id=str(event.parent_id) if event.parent_id else None,
        agent_id=event.agent_id,
        event_type=event.event_type.value
        if hasattr(event.event_type, "value")
        else str(event.event_type),
        status=event.status.value
        if hasattr(event.status, "value")
        else str(event.status),
        timestamp=event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp),
        input_data=event.input_data,
        output_data=event.output_data,
        latency_ms=event.latency_ms,
        metadata=event.metadata,
        children=children_map.get(eid, []) if children_map else [],
    )


def _compute_trace_summary(events: list[Any]) -> dict[str, Any]:
    """Compute status-based summary fields for a trace."""
    success_count = sum(
        1 for e in events
        if (e.status.value if hasattr(e.status, "value") else str(e.status)) == "success"
    )
    failure_count = sum(
        1 for e in events
        if (e.status.value if hasattr(e.status, "value") else str(e.status)) == "failure"
    )
    partial_count = sum(
        1 for e in events
        if (e.status.value if hasattr(e.status, "value") else str(e.status)) == "partial"
    )
    pending_count = sum(
        1 for e in events
        if (e.status.value if hasattr(e.status, "value") else str(e.status)) == "pending"
    )
    failure_event_ids = [
        str(e.event_id)
        for e in events
        if (e.status.value if hasattr(e.status, "value") else str(e.status)) == "failure"
    ]
    return {
        "success_count": success_count,
        "failure_count": failure_count,
        "partial_count": partial_count,
        "pending_count": pending_count,
        "failure_event_ids": failure_event_ids,
    }


def _extract_answer(trace: Trace) -> str:
    """Pull the final answer text from the trace events."""
    for event in reversed(trace.events):
        if (
            event.event_type == EventType.FINAL_RESPONSE
            and event.output_data
            and "answer" in event.output_data
        ):
            return event.output_data["answer"]
    # Fallback: look for the last LLM_CALL response.
    for event in reversed(trace.events):
        if (
            event.event_type == EventType.LLM_CALL
            and event.output_data
            and "response" in event.output_data
        ):
            return event.output_data["response"]
    return ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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

    children_map = _build_children_map(trace.events)
    return RunResponse(
        trace_id=str(trace.trace_id),
        response=_extract_answer(trace),
        events=[_serialize_event(e, children_map) for e in trace.events],
        total_latency_ms=trace.total_latency_ms,
        status="success",
    )


@app.get("/api/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    status: Optional[str] = Query(default=None, description="Filter events by status (all | success | failure | partial | pending)."),
    event_type: Optional[str] = Query(default=None, description="Filter events by event type (e.g. tool_call, llm_call)."),
) -> TraceResponse:
    """Retrieve a previously stored trace by its UUID, with optional filtering."""
    # --- Validate trace_id -------------------------------------------------
    try:
        tid = UUID(trace_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid trace_id format: {trace_id}",
        ) from exc

    # --- Validate filter values --------------------------------------------
    if status is not None and status not in _VALID_STATUS_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid status filter '{status}'. "
                f"Supported values: {sorted(_VALID_STATUS_FILTERS)}"
            ),
        )

    if event_type is not None and event_type not in _VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid event_type filter '{event_type}'. "
                f"Supported values: {sorted(_VALID_EVENT_TYPES)}"
            ),
        )

    # --- Fetch trace -------------------------------------------------------
    try:
        trace = await trace_store.get_trace(tid)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Trace {trace_id} not found",
        )

    all_events = trace.events

    # --- Build children map over the full event list -----------------------
    children_map = _build_children_map(all_events)

    # --- Compute summary over ALL events (before filtering) ----------------
    summary = _compute_trace_summary(all_events)

    # --- Apply filters -----------------------------------------------------
    filtered_events = list(all_events)

    if status is not None and status != "all":
        filtered_events = [
            e for e in filtered_events
            if (e.status.value if hasattr(e.status, "value") else str(e.status)) == status
        ]

    if event_type is not None:
        filtered_events = [
            e for e in filtered_events
            if (e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)) == event_type
        ]

    trace_status = (
        trace.status.value if hasattr(trace.status, "value") else str(trace.status)
    )

    return TraceResponse(
        trace_id=str(trace.trace_id),
        agent_id=trace.agent_id,
        status=trace_status,
        event_count=len(filtered_events),
        total_latency_ms=trace.total_latency_ms,
        success_count=summary["success_count"],
        failure_count=summary["failure_count"],
        partial_count=summary["partial_count"],
        pending_count=summary["pending_count"],
        failure_event_ids=summary["failure_event_ids"],
        events=[_serialize_event(e, children_map) for e in filtered_events],
    )
