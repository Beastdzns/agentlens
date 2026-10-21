#!/usr/bin/env python3
"""
Seed demo data into AgentLens so the trace-history panel has something to show
the very first time you open the dashboard.

By default this writes to ``./demo.db`` — a separate SQLite file so your
regular (memory-backed) runs are never polluted.  Point the app at the same
database to see the history:

    AGENTLENS_STORAGE=sqlite AGENTLENS_DB_PATH=./demo.db ./venv/bin/python -m uvicorn agentlens.web_api:app --reload

Then open ``http://localhost:8000/app/`` — the "Trace History" card lists the
seeded runs and clicking one loads its events, causal graph and root-cause
diagnosis.

Usage:
    ./venv/bin/python scripts/seed_demo_data.py
    # or, to use a different database file:
    AGENTLENS_DB_PATH=/tmp/demo.db ./venv/bin/python scripts/seed_demo_data.py
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    ObservationLevel,
    create_event,
    create_trace,
)
from agentlens.storage.db import close_database, init_db

DEMO_DB = os.getenv("AGENTLENS_DB_PATH", os.path.join(os.getcwd(), "demo.db"))
AGENT = "demo-research-agent"
BASE = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)


# ── Deterministic demo traces ──────────────────────────────────────────────

def demo_trace_1_success() -> tuple[list[AgentEvent], object]:
    """A clean 7-step research pipeline (the live-run shape from Member 1)."""
    trace_id = UUID("11111111-1111-1111-1111-111111111111")
    t = BASE
    plan = "1. Search current sources 2. Retrieve evidence 3. Synthesize findings"
    docs = [
        {"id": "research-paper-1", "relevance": 0.94},
        {"id": "research-paper-2", "relevance": 0.87},
        {"id": "technical-report-1", "relevance": 0.81},
    ]

    events: list[AgentEvent] = []
    e1 = create_event(trace_id, AGENT, EventType.LLM_CALL, EventStatus.SUCCESS,
                      {"query": "How can AI agents be debugged?"}, None, 25.0, None,
                      {"step": "planning"}, observation_level=ObservationLevel.FULL)
    e2 = create_event(trace_id, AGENT, EventType.DECISION, EventStatus.SUCCESS,
                      {"query": "How can AI agents be debugged?"}, {"plan": plan}, 120.0,
                      e1.event_id, {"step": "planning"})
    e3 = create_event(trace_id, AGENT, EventType.TOOL_CALL, EventStatus.SUCCESS,
                      {"tool": "web_search", "query": "AI agent debugging techniques"},
                      {"hits": ["paper-A", "paper-B", "paper-C"]}, 55.1, e2.event_id,
                      {"tool": "web_search", "step": "search"})
    e4 = create_event(trace_id, AGENT, EventType.RETRIEVAL, EventStatus.SUCCESS,
                      {"top_k": 3, "query": "observability patterns"},
                      {"documents": docs}, 38.4, e3.event_id, {"step": "retrieval"})
    e5 = create_event(trace_id, AGENT, EventType.MEMORY_WRITE, EventStatus.SUCCESS,
                      {"key": "retrieved_evidence"}, {"document_count": len(docs)}, 4.0,
                      e4.event_id, {"step": "memory-write"})
    e6 = create_event(trace_id, AGENT, EventType.LLM_CALL, EventStatus.SUCCESS,
                      {"prompt": "Synthesize findings about AI agent debugging"},
                      {"answer": "Structured observability with parent-linked traces "
                                 "enables fast root-cause localisation."}, 134.9, e5.event_id,
                      {"model": "gemini-2.5-flash", "tokens": 820, "step": "synthesis"})
    e7 = create_event(trace_id, AGENT, EventType.FINAL_RESPONSE, EventStatus.SUCCESS,
                      {}, {"answer": "AI agents can be debugged through structured "
                                     "observability: trace every LLM call, tool "
                                     "invocation, and decision with unique IDs and "
                                     "parent links."}, 1.0, e6.event_id,
                      {"step": "final-response"})

    events = [e1, e2, e3, e4, e5, e6, e7]
    trace = create_trace(AGENT, status=EventStatus.SUCCESS)
    trace.trace_id = trace_id
    trace.start_time = t
    trace.end_time = t + timedelta(milliseconds=sum(e.latency_ms or 0 for e in events))
    trace.events = events
    trace.event_count = len(events)
    trace.total_latency_ms = sum(e.latency_ms or 0 for e in events)
    return events, trace


def demo_trace_2_partial() -> tuple[list[AgentEvent], object]:
    """A tool call fails, a retry recovers — verdict "recovered"."""
    trace_id = UUID("22222222-2222-2222-2222-222222222222")
    t = BASE + timedelta(days=1)
    events: list[AgentEvent] = []
    e1 = create_event(trace_id, AGENT, EventType.LLM_CALL, EventStatus.SUCCESS,
                      {"prompt": "Research API error handling"}, {"text": "Plan: call /external"}, 212.0, None,
                      {"step": "planning"})
    e2 = create_event(trace_id, AGENT, EventType.TOOL_CALL, EventStatus.FAILURE,
                      {"tool": "api_call", "endpoint": "/external"},
                      {"error": "Connection timed out after 400 ms"}, 400.0, e1.event_id,
                      {"exception_type": "TimeoutError",
                       "exception_message": "Read timed out (connect=5, read=5)",
                       "step": "external-call"})
    e3 = create_event(trace_id, AGENT, EventType.RETRY, EventStatus.SUCCESS,
                      {"retry_count": 1}, {"status": "success on retry"}, 180.0, e2.event_id,
                      {"retry_of": str(e2.event_id), "attempt": 2})
    e4 = create_event(trace_id, AGENT, EventType.FINAL_RESPONSE, EventStatus.SUCCESS,
                      {}, {"answer": "Partial answer retrieved after retry."}, 99.2, e3.event_id,
                      {"step": "final-response"})
    events = [e1, e2, e3, e4]
    trace = create_trace(AGENT, status=EventStatus.PARTIAL)
    trace.trace_id = trace_id
    trace.start_time = t
    trace.end_time = t + timedelta(milliseconds=sum(e.latency_ms or 0 for e in events))
    trace.events = events
    trace.event_count = len(events)
    trace.total_latency_ms = sum(e.latency_ms or 0 for e in events)
    return events, trace


def demo_trace_3_failure() -> tuple[list[AgentEvent], object]:
    """An unrecoverable database failure — verdict "root_cause"."""
    trace_id = UUID("33333333-3333-3333-3333-333333333333")
    t = BASE + timedelta(days=2)
    events: list[AgentEvent] = []
    e1 = create_event(trace_id, AGENT, EventType.LLM_CALL, EventStatus.SUCCESS,
                      {"prompt": "Retrieve metrics from DB and summarise"},
                      {"text": "Planning DB query…"}, 300.0, None, {"step": "planning"})
    e2 = create_event(trace_id, AGENT, EventType.TOOL_CALL, EventStatus.FAILURE,
                      {"tool": "db_query", "query": "SELECT * FROM metrics"},
                      {"error": "Database connection refused"}, 550.0, e1.event_id,
                      {"exception_type": "ConnectionRefusedError",
                       "exception_message": "Connection refused on port 5432",
                       "step": "db-query"})
    e3 = create_event(trace_id, AGENT, EventType.ERROR, EventStatus.FAILURE,
                      {"context": "db failure"},
                      {"error": "Unrecoverable: cascading failure in data pipeline"},
                      2.5, e2.event_id,
                      {"exception_type": "RuntimeError", "step": "error-handler"})
    events = [e1, e2, e3]
    trace = create_trace(AGENT, status=EventStatus.FAILURE)
    trace.trace_id = trace_id
    trace.start_time = t
    trace.end_time = t + timedelta(milliseconds=sum(e.latency_ms or 0 for e in events))
    trace.events = events
    trace.event_count = len(events)
    trace.total_latency_ms = sum(e.latency_ms or 0 for e in events)
    return events, trace


def demo_trace_4_llm_failure() -> tuple[list[AgentEvent], object]:
    """A single failed planning LLM call — the trace Member 1 returns instead of 502."""
    trace_id = UUID("44444444-4444-4444-4444-444444444444")
    t = BASE + timedelta(days=3)
    e1 = create_event(trace_id, AGENT, EventType.LLM_CALL, EventStatus.FAILURE,
                      {"query": "Why did the deployment fail?"}, None, 110.0, None,
                      {"exception_type": "RuntimeError",
                       "exception_message": "Gemini quota exceeded",
                       "step": "planning"})
    events = [e1]
    trace = create_trace(AGENT, status=EventStatus.FAILURE)
    trace.trace_id = trace_id
    trace.start_time = t
    trace.end_time = t + timedelta(milliseconds=110.0)
    trace.events = events
    trace.event_count = 1
    trace.total_latency_ms = 110.0
    return events, trace


DEMO_TRACES = [
    ("Clean success run", demo_trace_1_success),
    ("Retry recovered a tool failure", demo_trace_2_partial),
    ("Database connection refused", demo_trace_3_failure),
    ("Gemini quota exceeded", demo_trace_4_llm_failure),
]


async def main() -> None:
    db = await init_db(DEMO_DB)

    from agentlens.storage.sqlite import SqliteEventStore, SqliteTraceStore

    events_store = SqliteEventStore(db)
    trace_store = SqliteTraceStore(db)

    total_events = 0
    for label, factory in DEMO_TRACES:
        events, trace = factory()
        await events_store.save_events_bulk(events)
        await trace_store.save_trace(trace)
        total_events += len(events)
        print(f"[seed] {label}: trace {trace.trace_id} — {len(events)} events, "
              f"status={trace.status.value}")

    count = await trace_store.count_traces()
    await close_database()
    print(f"\n[seed] done: {len(DEMO_TRACES)} traces, {total_events} events in {DEMO_DB}")
    print(f"[seed] run:  AGENTLENS_STORAGE=sqlite AGENTLENS_DB_PATH={DEMO_DB} "
          f"./venv/bin/python -m uvicorn agentlens.web_api:app --reload")
    print(f"[seed] then open: http://localhost:8000/app/")


if __name__ == "__main__":
    asyncio.run(main())
