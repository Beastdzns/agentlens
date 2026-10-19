"""
Benchmark for the SQLite storage backend (Phase 9, Member 3).

Run with::

    ./venv/bin/python tests/benchmark_storage.py

Not collected by pytest (no ``test_`` prefix).  Prints a markdown table used
in ``docs/phase9-reliability.md``.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import tempfile
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agentlens.events import AgentEvent, EventStatus, EventType, Trace
from agentlens.storage.db import Database, _run_migrations
from agentlens.storage.memory import InMemoryTraceStore
from agentlens.storage.sqlite import SqliteEventStore, SqliteTraceStore

BASE = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
N_EVENTS = 5000


def _event(trace_id, i):
    return AgentEvent(
        trace_id=trace_id,
        agent_id="bench-agent",
        event_type=EventType.LLM_CALL,
        timestamp=BASE + timedelta(microseconds=i),
        status=EventStatus.SUCCESS,
        input_data={"i": i, "text": "x" * 32},
        output_data={"ok": True},
        latency_ms=1.5,
    )


def _trace(trace_id, i):
    return Trace(
        trace_id=trace_id,
        agent_id="bench-agent",
        status=EventStatus.SUCCESS,
        start_time=BASE + timedelta(seconds=i),
        end_time=BASE + timedelta(seconds=i + 1),
        event_count=5,
        total_latency_ms=7.5,
    )


async def _bench_event_writes(event_store):
    # Individual saves.
    per_event_trace = uuid4()
    start = time.perf_counter()
    for i in range(N_EVENTS):
        await event_store.save_event(_event(per_event_trace, i))
    individual = time.perf_counter() - start

    # Bulk save.
    bulk_trace = uuid4()
    events = [_event(bulk_trace, i) for i in range(N_EVENTS)]
    start = time.perf_counter()
    await event_store.save_events_bulk(events)
    bulk = time.perf_counter() - start

    return individual, bulk


async def _bench_list(sqlite_store, n_traces, queries=50, limit=20):
    for i in range(n_traces):
        await sqlite_store.save_trace(_trace(uuid4(), i))

    latencies = []
    for _ in range(queries):
        start = time.perf_counter()
        await sqlite_store.list_trace_page(limit=limit)
        latencies.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    await sqlite_store.count_traces()
    count_ms = (time.perf_counter() - start) * 1000

    return statistics.mean(latencies), statistics.median(latencies), count_ms


async def _bench_list_memory(n_traces, queries=50, limit=20):
    store = InMemoryTraceStore()
    for i in range(n_traces):
        await store.save_trace(_trace(uuid4(), i))

    latencies = []
    for _ in range(queries):
        start = time.perf_counter()
        await store.list_trace_page(limit=limit)
        latencies.append((time.perf_counter() - start) * 1000)
    return statistics.mean(latencies), statistics.median(latencies)


async def main() -> None:
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench.db")
    db = Database(db_path)
    await db._open()
    await _run_migrations(db)
    event_store = SqliteEventStore(db)
    trace_store = SqliteTraceStore(db)

    print("\n## Benchmark results\n")
    print(f"- SQLite file: {db_path}")
    print(f"- Events per throughput test: {N_EVENTS}\n")

    individual, bulk = await _bench_event_writes(event_store)
    print("### Event insert throughput\n")
    print("| Method | Total (s) | Events/s |")
    print("|---|---|---|")
    print(f"| `save_event` (per event) | {individual:.3f} | {N_EVENTS/individual:,.0f} |")
    print(f"| `save_events_bulk` (one tx) | {bulk:.4f} | {N_EVENTS/bulk:,.0f} |")
    print(f"| Speed-up | | {individual/bulk:,.1f}x |")

    print("\n### `list_trace_page(limit=20)` latency (50 calls)\n")
    print("| Backend | Traces | Mean (ms) | Median (ms) | `count_traces` (ms) |")
    print("|---|---|---|---|---|")
    for n in (1000, 10000):
        mean, median, count_ms = await _bench_list(trace_store, n)
        print(f"| SQLite | {n:,} | {mean:.2f} | {median:.2f} | {count_ms:.2f} |")
        m_mean, m_median = await _bench_list_memory(n)
        print(f"| In-memory | {n:,} | {m_mean:.2f} | {m_median:.2f} | — |")

    await db.close()
    print()


if __name__ == "__main__":
    asyncio.run(main())
