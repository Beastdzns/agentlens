"""
Phase 10 benchmark: full tracing vs adaptive tracing.

Run it directly::

    ./venv/bin/python tests/benchmark_observation.py

It builds one synthetic trace (a deep, repetitive agent run that ends in a
terminal failure), processes it through both observation modes, and reports:

- **event volume**    — how many events are retained and at which level;
- **payload size**    — bytes of retained input/output/metadata;
- **storage usage**   — on-disk size of a SQLite database holding the result;
- **latency overhead**— wall-clock cost of scoring + policy application;
- **diagnosis**       — whether the failure evidence survived observation.

The numbers printed here are the ones quoted in
``docs/phase10-adaptive-observability.md``.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agentlens.events import AgentEvent, EventStatus, EventType
from agentlens.observation import ImportanceScorer, ObservationPolicy, PolicyConfig
from agentlens.storage import SqliteEventStore, init_db
from agentlens.storage.db import close_database

EVENT_COUNT = 10_000


def generate_synthetic_trace(event_count: int = EVENT_COUNT) -> list[AgentEvent]:
    """A repetitive run that ends in a terminal failure."""
    events: list[AgentEvent] = []
    trace_id = str(uuid4())
    for i in range(event_count):
        is_failure = i == event_count - 1
        events.append(
            AgentEvent(
                event_id=str(uuid4()),
                trace_id=trace_id,
                parent_id=None if i == 0 else events[0].event_id,
                agent_id="benchmark-agent",
                event_type=EventType.ERROR if is_failure else EventType.LLM_CALL,
                status=EventStatus.FAILURE if is_failure else EventStatus.SUCCESS,
                timestamp=datetime.now(timezone.utc),
                input_data={"context": "A" * 500, "query": f"iter {i}"},
                output_data={"raw": "B" * 500, "parsed": {"status": "ok"}},
                latency_ms=100.0,
                metadata={"exception_type": "TimeoutError"} if is_failure else {},
            )
        )
    return events


def _payload_bytes(event: AgentEvent) -> int:
    return len(
        json.dumps(
            {
                "input": event.input_data,
                "output": event.output_data,
                "metadata": event.metadata,
            },
            default=str,
        ).encode()
    )


async def _sqlite_bytes(events: list[AgentEvent]) -> int:
    """Bytes on disk for a SQLite database holding *events*."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "benchmark.db"
        db = await init_db(db_path)
        try:
            store = SqliteEventStore(db)
            for event in events:
                await store.save_event(event)
        finally:
            await close_database()
        return db_path.stat().st_size


async def run_benchmark(mode: str, events: list[AgentEvent]) -> dict:
    """Process *events* under *mode* and collect the measurements."""
    scorer = ImportanceScorer()
    policy = ObservationPolicy(PolicyConfig(mode=mode))

    start = time.perf_counter()
    transformed = []
    total_bytes = 0
    levels: dict[str, int] = {}

    for i, event in enumerate(events):
        repeat_index = i if event.event_type == EventType.LLM_CALL else 0
        score = scorer.score(event, repeat_index=repeat_index)
        level = policy.decide(score, event)
        final_event = policy.transform(event, level)
        transformed.append(final_event)
        total_bytes += _payload_bytes(final_event)
        levels[level.value] = levels.get(level.value, 0) + 1

    duration = time.perf_counter() - start
    sqlite_bytes = await _sqlite_bytes(transformed)

    failure = transformed[-1]
    diagnosis_preserved = (
        failure.status == EventStatus.FAILURE
        and len(str(failure.input_data)) > 100
        and "exception_type" in (failure.metadata or {})
    )

    return {
        "mode": mode,
        "events": len(transformed),
        "levels": levels,
        "payload_bytes": total_bytes,
        "sqlite_bytes": sqlite_bytes,
        "duration": duration,
        "diagnosis_preserved": diagnosis_preserved,
    }


async def main() -> None:
    print(f"Generating {EVENT_COUNT:,} synthetic events...")
    events = generate_synthetic_trace(EVENT_COUNT)

    results = []
    for mode in ("full", "adaptive"):
        results.append(await run_benchmark(mode, events))

    full, adaptive = results

    print()
    print(f"{'Metric':<28}{'Full':>18}{'Adaptive':>18}{'Change':>14}")
    print("-" * 78)
    print(
        f"{'Events retained':<28}{full['events']:>18,}{adaptive['events']:>18,}"
        f"{'same':>14}"
    )
    print(
        f"{'Payload size (KB)':<28}{full['payload_bytes'] / 1024:>18,.1f}"
        f"{adaptive['payload_bytes'] / 1024:>18,.1f}"
        f"{_delta(full['payload_bytes'], adaptive['payload_bytes']):>14}"
    )
    print(
        f"{'SQLite file (KB)':<28}{full['sqlite_bytes'] / 1024:>18,.1f}"
        f"{adaptive['sqlite_bytes'] / 1024:>18,.1f}"
        f"{_delta(full['sqlite_bytes'], adaptive['sqlite_bytes']):>14}"
    )
    print(
        f"{'Processing time (s)':<28}{full['duration']:>18.4f}{adaptive['duration']:>18.4f}"
        f"{_delta(full['duration'], adaptive['duration']):>14}"
    )
    print()
    print("Observation level distribution:")
    for mode_result in results:
        print(f"  {mode_result['mode']:<10}{mode_result['levels']}")
    print()
    print(f"Diagnosis preserved (full)    : {full['diagnosis_preserved']}")
    print(f"Diagnosis preserved (adaptive): {adaptive['diagnosis_preserved']}")


def _delta(baseline: float, candidate: float) -> str:
    if baseline == 0:
        return "n/a"
    change = (candidate - baseline) / baseline * 100
    return f"{change:+.1f}%"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
