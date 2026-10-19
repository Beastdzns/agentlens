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
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agentlens.events import EventStatus, EventType, Trace
from agentlens.sdk_collector import EventCollector
from contextlib import asynccontextmanager
from agentlens.storage import (
    EventStore,
    TraceStore,
    InMemoryEventStore,
    InMemoryTraceStore,
    SqliteEventStore,
    SqliteTraceStore,
    init_db,
    get_database,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

def _backend() -> str:
    """Return the configured storage backend ('memory' | 'sqlite')."""
    return os.getenv("AGENTLENS_STORAGE", "memory").lower()


@asynccontextmanager
async def lifespan(app):  # noqa: ANN001
    """Lifespan: wire up SQLite stores when AGENTLENS_STORAGE=sqlite."""
    global event_store, trace_store
    if _backend() == "sqlite":
        db = await init_db()
        event_store, trace_store = SqliteEventStore(db), SqliteTraceStore(db)
    yield
    if _backend() == "sqlite":
        await get_database().close()


app = FastAPI(
    title="AgentLens API",
    description="HTTP interface for the AgentLens multi-step research agent.",
    version="0.1.0",
    lifespan=lifespan,
)

def _resolve_frontend_dir() -> Path | None:
    """Locate the bundled ``frontend/`` dashboard directory.

    Tries the src-layout location first (``<repo-root>/frontend`` relative
    to this file), then falls back to the current working directory so the
    dashboard still mounts when the package is installed non-editable or
    the server is launched from the repo root.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "frontend",
        Path(__file__).resolve().parents[1] / "frontend",
        Path.cwd() / "frontend",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "index.html").is_file():
            return candidate
    return None


_frontend_dir = _resolve_frontend_dir()
if _frontend_dir is not None:
    app.mount(
        "/app",
        StaticFiles(directory=str(_frontend_dir), html=True),
        name="frontend",
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
# Health check & root redirect
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint for frontend/API monitoring."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def root_redirect() -> dict[str, str]:
    """Redirect root to the dashboard."""
    return {"message": "AgentLens API", "dashboard": "/app/", "docs": "/docs"}


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



class TraceSummary(BaseModel):
    """Lightweight summary returned by ``GET /api/traces``."""

    trace_id: str
    agent_id: Optional[str] = None
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    event_count: int = 0
    total_latency_ms: float = 0.0
    metadata: Optional[dict[str, Any]] = None


class TraceListResponse(BaseModel):
    """Body returned by ``GET /api/traces``."""

    items: list[TraceSummary]
    count: int
    next_cursor: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent runner (encapsulates the multi-step workflow)
# ---------------------------------------------------------------------------


class _AgentRunner:
    """Runs the multi-step research agent and returns a completed Trace."""

    def __init__(
        self,
        ev_store: EventStore,
        tr_store: TraceStore,
    ) -> None:
        self._event_store = ev_store
        self._trace_store = tr_store

    async def run(self, query: str) -> Trace:
        """Execute the full pipeline and return a finalised Trace.

        Each of the six pipeline steps is wrapped in a per-step capture context
        manager (``capture_llm_call``, ``capture_tool_call``, ``capture_step``).

        * On **success** the wrapper records a ``SUCCESS`` event before yielding
          back to this method.
        * On **any exception** the wrapper records a ``FAILURE`` event carrying
          ``exception_type`` and ``exception_message`` in its metadata **before
          re-raising**, so the trace always contains a persisted failure record
          even if the process is interrupted mid-run.

        The single outer ``except`` block is now responsible *only* for calling
        ``end_trace`` and persisting the trace as ``FAILURE`` — the per-step
        wrappers handle the granular failure recording.

        **Retry linkage**: when retrying a step after a failure, pass
        ``retry_of=<failed_event_id>`` and an incremented ``attempt`` counter to
        the wrapper.  Both values are stored in the new event's metadata so the
        UI can trace the retry chain.
        """
        collector = EventCollector(storage_backend=self._event_store)
        # --- Initialise Gemini (if key is available) -----------------------
        gemini_client: Any = None
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if api_key:
            try:
                from google import genai

                gemini_client = genai.Client(api_key=api_key)
            except ImportError:
                pass  # dependency not installed — use fallback

        async def _ask_gemini(prompt: str, fallback: str) -> str:
            if gemini_client is None:
                return fallback
            try:
                resp = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                )
                return resp.text
            except Exception:
                raise  # let caller (step wrapper) handle Gemini failures

        # --- Start trace ---------------------------------------------------
        trace_id = await collector.start_trace(agent_id="multistep-research-agent")
        started = time.perf_counter()
        answer = ""

        try:
            # ------------------------------------------------------------------
            # Step 1 — Planning
            # capture_llm_call times the call, records SUCCESS on return, and
            # records FAILURE with exception_type + exception_message on error.
            # ------------------------------------------------------------------
            planning_prompt = f"Create a concise three-step research plan for: {query}"
            async with collector.capture_llm_call(
                prompt=planning_prompt,
                model=model_name,
                metadata={"step": "planning"},
            ) as planning_result:
                planning_result[0] = await _ask_gemini(
                    planning_prompt,
                    "1. Search current sources 2. Retrieve useful evidence 3. Synthesize findings",
                )
            plan = planning_result[0]

            # Record the planning decision event (with the plan as output_data)
            # used for parent-linkage in subsequent steps.
            planning_event = await collector.record_event(
                event_type=EventType.DECISION,
                input_data={"query": query},
                output_data={"plan": plan},
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata={"step": "planning"},
            )

            # ------------------------------------------------------------------
            # Step 2 — Web search (simulated tool call)
            # capture_tool_call records SUCCESS with input+output, or FAILURE
            # with exception metadata on error, then re-raises.
            # ------------------------------------------------------------------
            collector.push_parent(planning_event.event_id)
            async with collector.capture_tool_call(
                "web_search",
                input_data={"tool": "web_search", "query": query},
                metadata={"step": "search"},
            ) as search_result:
                search_result[0] = {
                    "hits": [
                        "research-paper-1",
                        "research-paper-2",
                        "technical-report-1",
                    ]
                }

            # Record a dedicated TOOL_CALL event for parent-linkage purposes.
            search_event = await collector.record_event(
                event_type=EventType.TOOL_CALL,
                input_data={"tool": "web_search", "query": query},
                output_data=search_result[0],
                latency_ms=0.0,
                metadata={"step": "search", "tool": "web_search"},
            )

            # ------------------------------------------------------------------
            # Step 3 — Retrieval
            # ------------------------------------------------------------------
            collector.push_parent(search_event.event_id)
            retrieval_docs: dict[str, Any] = {
                "documents": [
                    {"id": "research-paper-1", "relevance": 0.94},
                    {"id": "research-paper-2", "relevance": 0.87},
                    {"id": "technical-report-1", "relevance": 0.81},
                ]
            }
            async with collector.capture_step(
                EventType.RETRIEVAL,
                input_data={"top_k": 3, "query": query},
                metadata={"step": "retrieval"},
            ):
                pass  # Simulated retrieval — no external I/O

            retrieval_event = await collector.record_event(
                event_type=EventType.RETRIEVAL,
                input_data={"top_k": 3, "query": query},
                output_data=retrieval_docs,
                latency_ms=0.0,
                metadata={"step": "retrieval"},
            )
            collector.pop_parent()

            # ------------------------------------------------------------------
            # Step 4 — Memory write
            # ------------------------------------------------------------------
            async with collector.capture_step(
                EventType.MEMORY_WRITE,
                input_data={"key": "retrieved_evidence"},
                metadata={"step": "memory-write"},
            ):
                pass  # In-memory store, no external I/O

            await collector.record_event(
                event_type=EventType.MEMORY_WRITE,
                input_data={"key": "retrieved_evidence"},
                output_data={
                    "document_count": len(retrieval_event.output_data or {})
                },
                latency_ms=2.0,
                metadata={"step": "memory-write"},
            )
            collector.pop_parent()

            # ------------------------------------------------------------------
            # Step 5 — Synthesis LLM call
            # If Gemini raises here, capture_llm_call records a FAILURE event
            # with exception_type + exception_message before re-raising so the
            # outer except only has to persist/finalise the trace.
            #
            # Retry example (not triggered in the default path):
            #   If a previous synthesis attempt failed and we stored its
            #   event_id as `failed_synthesis_id`, we would call:
            #       collector.capture_llm_call(
            #           prompt=..., retry_of=failed_synthesis_id, attempt=2
            #       )
            # ------------------------------------------------------------------
            synthesis_prompt = (
                f"Answer this query using the plan and evidence below:\n"
                f"Query: {query}\nPlan: {plan}\n"
                f"Evidence: {retrieval_event.output_data}"
            )
            async with collector.capture_llm_call(
                prompt=query,
                model=model_name,
                metadata={"step": "synthesis"},
            ) as synthesis_result:
                synthesis_result[0] = await _ask_gemini(
                    synthesis_prompt,
                    "The evidence suggests a measurable, traceable workflow with "
                    "planning, retrieval, and synthesis.",
                )
            answer = synthesis_result[0]

            # ------------------------------------------------------------------
            # Step 6 — Final response
            # ------------------------------------------------------------------
            async with collector.capture_step(
                EventType.FINAL_RESPONSE,
                metadata={"step": "final-response"},
            ):
                pass  # No work; the wrapper records the SUCCESS bookend

            await collector.record_event(
                event_type=EventType.FINAL_RESPONSE,
                output_data={"answer": answer},
                latency_ms=0.0,
                metadata={"step": "final-response"},
            )

            await collector.end_trace()

        except Exception:
            # Each per-step wrapper already recorded a FAILURE event (with
            # exception_type + exception_message) before re-raising, so the
            # trace contains exactly one FAILURE event for the broken step.
            # We only need to finalise the trace and persist it as FAILURE.
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


@app.get("/api/traces", response_model=TraceListResponse, tags=["traces"])
async def list_traces(
    agent_id: Optional[str] = Query(default=None, description="Filter by agent ID."),
    status: Optional[str] = Query(
        default=None,
        description="Filter by status (all | success | failure | partial | pending).",
    ),
    start_date: Optional[str] = Query(
        default=None,
        description="Earliest start_time (YYYY-MM-DD or ISO-8601).",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="Latest start_time (YYYY-MM-DD or ISO-8601).",
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Page size."),
    cursor: Optional[str] = Query(
        default=None,
        description="Keyset pagination cursor from a previous response.",
    ),
) -> TraceListResponse:
    """Return a paginated, filtered list of stored traces (no event payloads)."""
    from datetime import datetime, timezone

    # -- Validate status
    status_enum = None
    if status is not None and status != "all":
        if status not in _VALID_STATUS_FILTERS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid status filter {status!r}. "
                    f"Supported values: {sorted(_VALID_STATUS_FILTERS)}"
                ),
            )
        try:
            status_enum = EventStatus(status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}") from exc

    # -- Parse dates
    def _parse_date(raw: str) -> datetime:
        """Accept YYYY-MM-DD or ISO-8601; return UTC-aware datetime."""
        try:
            if len(raw) == 10:
                return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format {raw!r}. Use YYYY-MM-DD or ISO-8601.",
            ) from exc

    start_dt = _parse_date(start_date) if start_date is not None else None
    end_dt = _parse_date(end_date) if end_date is not None else None

    # -- Early cursor validation
    if cursor is not None:
        import base64 as _b64
        try:
            decoded = _b64.urlsafe_b64decode(cursor.encode()).decode()
            if "|" not in decoded:
                raise ValueError("missing separator")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Malformed pagination cursor.") from exc

    # -- Storage calls
    try:
        items, next_cursor = await trace_store.list_trace_page(
            limit=limit,
            agent_id=agent_id,
            status=status_enum,
            start_date=start_dt,
            end_date=end_dt,
            cursor=cursor,
        )
        total = await trace_store.count_traces(
            agent_id=agent_id,
            status=status_enum,
            start_date=start_dt,
            end_date=end_dt,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage error: {exc}") from exc

    # -- Build summaries
    def _to_summary(t: Any) -> TraceSummary:
        st = t.status.value if hasattr(t.status, "value") else str(t.status)
        start = t.start_time.isoformat() if t.start_time else None
        end = t.end_time.isoformat() if t.end_time else None
        return TraceSummary(
            trace_id=str(t.trace_id),
            agent_id=t.agent_id,
            status=st,
            start_time=start,
            end_time=end,
            event_count=getattr(t, "event_count", 0),
            total_latency_ms=t.total_latency_ms or 0.0,
            metadata=getattr(t, "metadata", None),
        )

    return TraceListResponse(
        items=[_to_summary(t) for t in items],
        count=total,
        next_cursor=next_cursor,
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
