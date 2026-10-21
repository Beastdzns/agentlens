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
import logging
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
    close_database,
)
from agentlens.observation import build_policy_and_scorer
from agentlens.causal import build_graph, RootCauseAnalyzer

load_dotenv()

logger = logging.getLogger("agentlens.web_api")


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
        # Close *and reset* the singleton so a later start (or the next test)
        # re-opens a fresh connection instead of reusing the closed one, then
        # restore the shared default instances (preserving their identity).
        await close_database()
        event_store, trace_store = _default_event_store, _default_trace_store


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

# The shared default singletons.  The lifespan swaps the globals to SQLite
# stores while running and restores *these exact objects* on shutdown, so
# modules that imported ``event_store`` / ``trace_store`` at import time keep
# a valid reference (identity must be preserved across start/stop cycles).
_default_event_store = event_store
_default_trace_store = trace_store

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
    importance_score: float = 0.5
    observation_level: str = "full"
    # Derived convenience field — list of direct children event IDs.
    children: list[str] = Field(default_factory=list)


class RunResponse(BaseModel):
    """Body returned by ``POST /api/run``."""

    trace_id: str
    response: str
    events: list[SerializedEvent]
    total_latency_ms: float
    status: str
    # Optional observation level summary (populated in adaptive mode).
    observation_summary: Optional[dict[str, int]] = None
    # Set when the run failed.  The trace is still returned so the dashboard
    # can show the causal analysis of the failure.
    error: Optional[str] = None
    # Optional causal graph (Phase 11).  Additive, same rules as the trace
    # response: omitted when the graph cannot be built (fail-open, D8).
    causal_graph: Optional["CausalGraphResponse"] = None
    # Optional root-cause diagnosis (Phase 11).  Additive, same rules.
    root_cause: Optional["RootCauseResponse"] = None


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
    # Optional causal graph (Phase 11).  Additive: ``None`` when the trace has
    # no events or when graph construction failed (fail-open, D8).
    causal_graph: Optional["CausalGraphResponse"] = None
    # Optional root-cause diagnosis (Phase 11).  Additive, same rules.
    root_cause: Optional["RootCauseResponse"] = None


class CausalNodeModel(BaseModel):
    """One event rendered as a node of the causal graph."""

    event_id: str
    parent_id: Optional[str] = None
    event_type: str
    status: str
    timestamp: str
    latency_ms: Optional[float] = None
    importance_score: float = 0.5
    observation_level: str = "full"
    label: str
    error_message: Optional[str] = None
    exception_type: Optional[str] = None
    is_failure: bool = False
    suspicious_output: bool = False


class CausalEdgeModel(BaseModel):
    """A directed relation between two causal nodes."""

    source: str
    target: str
    relation: str


class CausalGraphResponse(BaseModel):
    """Body returned by ``GET /api/traces/{trace_id}/causal-graph``."""

    trace_id: str
    nodes: list[CausalNodeModel] = Field(default_factory=list)
    edges: list[CausalEdgeModel] = Field(default_factory=list)


class RootCauseResponse(BaseModel):
    """Body returned by ``GET /api/traces/{trace_id}/root-cause``."""

    trace_id: str
    verdict: str = "no_failure"
    root_cause_event_id: Optional[str] = None
    confidence: float = 0.0
    category: str = "none"
    explanation: str = ""
    evidence_path: list[str] = Field(default_factory=list)
    primary_failures: list[str] = Field(default_factory=list)
    secondary_failures: list[str] = Field(default_factory=list)
    affected_events: list[str] = Field(default_factory=list)
    recovered_failures: list[str] = Field(default_factory=list)


# Resolve the ``causal_graph`` forward reference declared on ``TraceResponse``.
TraceResponse.model_rebuild()


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


class AgentRunFailed(Exception):
    """
    Raised when the agent pipeline fails *inside* a traced run.

    The failed trace has already been persisted when this is raised, so the
    exception carries its id: the API can then return the trace (with its
    causal graph and root-cause diagnosis) instead of throwing the run away.
    """

    def __init__(self, trace_id: str, cause: BaseException) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.trace_id = trace_id
        self.cause = cause


class _AgentRunner:
    """Runs the multi-step research agent and returns a completed Trace."""

    def __init__(
        self,
        ev_store: EventStore,
        tr_store: TraceStore,
    ) -> None:
        self._event_store = ev_store
        self._trace_store = tr_store
        # Build scorer + policy from env (D8: adaptive is opt-in).
        self._scorer, self._policy = build_policy_and_scorer()

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
        wrappers handle the granular failure recording.  It then raises
        :class:`AgentRunFailed`, which carries the id of the persisted failed
        trace so the caller can return it instead of losing the run.

        **Retry linkage**: when retrying a step after a failure, pass
        ``retry_of=<failed_event_id>`` and an incremented ``attempt`` counter to
        the wrapper.  Both values are stored in the new event's metadata so the
        UI can trace the retry chain.

        Raises:
            AgentRunFailed: when a pipeline step fails.  The failed trace has
                already been persisted when this is raised.
        """
        collector = EventCollector(storage_backend=self._event_store,
                                   scorer=self._scorer,
                                   policy=self._policy)
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

            # The capture wrapper already recorded the TOOL_CALL event — with
            # the real timing and the output.  Reuse it instead of recording a
            # second, duplicate event for the same step.
            search_event = collector.last_event

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
            ) as retrieval_result:
                # Simulated retrieval — no external I/O.
                retrieval_result[0] = retrieval_docs

            retrieval_event = collector.last_event
            collector.pop_parent()

            # ------------------------------------------------------------------
            # Step 4 — Memory write
            # ------------------------------------------------------------------
            collector.push_parent(retrieval_event.event_id)
            async with collector.capture_step(
                EventType.MEMORY_WRITE,
                input_data={"key": "retrieved_evidence"},
                metadata={"step": "memory-write"},
            ) as memory_result:
                # In-memory store, no external I/O.
                memory_result[0] = {
                    "document_count": len(
                        (retrieval_event.output_data or {}).get("documents", [])
                    )
                }
            memory_event = collector.last_event
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
            collector.push_parent(memory_event.event_id)
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
            synthesis_event = collector.last_event
            collector.pop_parent()

            # ------------------------------------------------------------------
            # Step 6 — Final response
            # ------------------------------------------------------------------
            collector.push_parent(synthesis_event.event_id)
            async with collector.capture_step(
                EventType.FINAL_RESPONSE,
                metadata={"step": "final-response"},
            ) as final_result:
                # No work; the wrapper records the SUCCESS bookend.
                final_result[0] = {"answer": answer}
            collector.pop_parent()

            await collector.end_trace()

        except Exception as exc:
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
            # The failed trace is persisted — tell the caller which trace to
            # look at instead of losing the whole run.
            raise AgentRunFailed(str(trace_id), exc) from exc

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
    # Safely extract observation fields (may be absent on legacy events).
    importance_score = getattr(event, "importance_score", 0.5)
    obs_level_raw = getattr(event, "observation_level", "full")
    observation_level = (
        obs_level_raw.value if hasattr(obs_level_raw, "value") else str(obs_level_raw)
    )
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
        importance_score=importance_score,
        observation_level=observation_level,
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


def _build_causal_graph_response(
    trace_id: str, events: list[Any]
) -> Optional[CausalGraphResponse]:
    """
    Build the causal-graph payload for a trace.

    Fail-open (D8): a graph that cannot be built must never turn a trace
    request into a 5xx.  On error the graph is simply omitted (``None``)
    and the failure is logged.
    """
    try:
        graph = build_graph(events)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("Causal graph construction failed for trace %s: %s", trace_id, exc)
        return None

    return CausalGraphResponse(
        trace_id=str(trace_id),
        nodes=[CausalNodeModel(**node.to_dict()) for node in graph.nodes.values()],
        edges=[CausalEdgeModel(**edge.to_dict()) for edge in graph.edges],
    )


def _analyze_root_cause(
    trace_id: str, events: list[Any]
) -> Optional[RootCauseResponse]:
    """
    Diagnose the root cause of a trace.

    Fail-open (D8): a diagnosis that cannot be produced must never turn a
    trace request into a 5xx.  On error the payload is omitted (``None``)
    and the failure is logged.
    """
    try:
        graph = build_graph(events)
        diagnosis = RootCauseAnalyzer().analyze(graph)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("Root-cause analysis failed for trace %s: %s", trace_id, exc)
        return None

    return RootCauseResponse(trace_id=str(trace_id), **diagnosis.to_dict())


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
    """
    Run the multi-step research agent and return the complete trace.

    A run that fails *inside* the pipeline is **not** an HTTP error: the
    failed trace is persisted by the runner, so it is returned like any other
    run — with ``status="failure"``, the error message and the causal
    analysis.  That is what lets the dashboard show *why* a run failed
    instead of a bare "something went wrong".

    Only a failure of the run *machinery* itself (no trace, no store) is
    reported as ``502``.
    """
    runner = _AgentRunner(event_store, trace_store)

    error: Optional[str] = None
    try:
        trace = await runner.run(body.query)
    except AgentRunFailed as exc:
        try:
            trace = await trace_store.get_trace(UUID(exc.trace_id))
        except KeyError:
            raise HTTPException(
                status_code=502,
                detail=f"Agent execution failed: {exc}",
            ) from exc
        error = str(exc)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agent execution failed: {exc}",
        ) from exc

    children_map = _build_children_map(trace.events)
    serialized = [_serialize_event(e, children_map) for e in trace.events]

    # Build per-level counts (only meaningful in adaptive mode but always populated).
    obs_summary: dict[str, int] = {}
    for ev in serialized:
        obs_summary[ev.observation_level] = obs_summary.get(ev.observation_level, 0) + 1

    trace_id = str(trace.trace_id)
    return RunResponse(
        trace_id=trace_id,
        response=_extract_answer(trace),
        events=serialized,
        total_latency_ms=trace.total_latency_ms,
        status="failure" if trace.status == EventStatus.FAILURE else "success",
        observation_summary=obs_summary,
        error=error,
        causal_graph=_build_causal_graph_response(trace_id, trace.events),
        root_cause=_analyze_root_cause(trace_id, trace.events),
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
    from datetime import date as _date, time as _time

    def _parse_date(raw: str, *, end_of_day: bool = False) -> datetime:
        """Accept YYYY-MM-DD or ISO-8601; return a UTC-aware datetime.

        A date-only value covers the whole day: ``start_date`` is anchored to
        ``00:00:00`` and ``end_date`` to ``23:59:59.999999``.  This keeps
        ``end_date=<today>`` inclusive of traces that started later today.
        """
        try:
            if len(raw) == 10:
                day = _date.fromisoformat(raw)
                clock = _time(23, 59, 59, 999999) if end_of_day else _time(0, 0, 0)
                return datetime.combine(day, clock, tzinfo=timezone.utc)
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format {raw!r}. Use YYYY-MM-DD or ISO-8601.",
            ) from exc

    start_dt = _parse_date(start_date) if start_date is not None else None
    end_dt = _parse_date(end_date, end_of_day=True) if end_date is not None else None

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
    include_causal_graph: bool = Query(default=True, description="Include the Phase 11 causal graph in the response."),
    include_root_cause: bool = Query(default=True, description="Include the Phase 11 root-cause diagnosis in the response."),
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

    causal_graph = (
        _build_causal_graph_response(str(trace.trace_id), all_events)
        if include_causal_graph
        else None
    )
    root_cause = (
        _analyze_root_cause(str(trace.trace_id), all_events)
        if include_root_cause
        else None
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
        causal_graph=causal_graph,
        root_cause=root_cause,
    )


@app.get(
    "/api/traces/{trace_id}/causal-graph",
    response_model=CausalGraphResponse,
    tags=["traces"],
)
async def get_causal_graph(trace_id: str) -> CausalGraphResponse:
    """
    Return the causal graph of a stored trace.

    The graph is built from **all** events of the trace — filters do not
    change the causal structure.  An empty trace yields an empty graph
    rather than an error, and a graph that cannot be built degrades to an
    empty graph (fail-open, D8).
    """
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

    graph = _build_causal_graph_response(str(trace.trace_id), trace.events)
    if graph is None:
        return CausalGraphResponse(trace_id=str(trace.trace_id))
    return graph


@app.get(
    "/api/traces/{trace_id}/root-cause",
    response_model=RootCauseResponse,
    tags=["traces"],
)
async def get_root_cause(trace_id: str) -> RootCauseResponse:
    """
    Diagnose what caused the failure of a stored trace.

    The diagnosis is rule-based and grounded in recorded evidence: it never
    calls a model and never invents a cause.  A trace without failures yields
    ``verdict="no_failure"`` rather than a guess.
    """
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

    diagnosis = _analyze_root_cause(str(trace.trace_id), trace.events)
    if diagnosis is None:
        return RootCauseResponse(trace_id=str(trace.trace_id))
    return diagnosis
