"""
Quick smoke-test for the SQLite storage backend.
Run with:  python tests/smoke_sqlite.py
"""
from __future__ import annotations

import asyncio
import tempfile
import os
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from agentlens.events import AgentEvent, EventType, EventStatus, Trace, ObservationLevel
from agentlens.storage.db import Database, _run_migrations
from agentlens.storage.sqlite import SqliteEventStore, SqliteTraceStore


async def main() -> None:
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    db = Database(db_path)
    try:
        await db._open()
        await _run_migrations(db)

        event_store = SqliteEventStore(db)
        trace_store = SqliteTraceStore(db)

        # -- basic event round-trip -----------------------------------------
        trace_id = uuid4()
        event = AgentEvent(
            trace_id=trace_id,
            agent_id="smoke-agent",
            event_type=EventType.LLM_CALL,
            status=EventStatus.SUCCESS,
            input_data={"prompt": "hello"},
            output_data={"text": "world"},
            metadata={"model": "gemini-pro"},
            latency_ms=42.5,
            parent_id=None,
            importance_score=0.9,
            observation_level=ObservationLevel.FULL,
        )
        await event_store.save_event(event)
        loaded = await event_store.get_event(event.event_id)
        assert loaded.event_id == event.event_id, "event_id mismatch"
        assert loaded.input_data == {"prompt": "hello"}, "input_data mismatch"
        assert loaded.metadata == {"model": "gemini-pro"}, "metadata mismatch"
        assert loaded.parent_id is None, "parent_id should be None"
        print("[OK] Event round-trip")

        # -- get_trace_events ordering ----------------------------------------
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = []
        for i in range(5):
            e = AgentEvent(
                trace_id=trace_id,
                agent_id="smoke-agent",
                event_type=EventType.TOOL_CALL,
                status=EventStatus.SUCCESS,
                timestamp=t0 + timedelta(seconds=i * 10),
            )
            events.append(e)
        # Save in reverse
        await event_store.save_events_bulk(list(reversed(events)))
        result = await event_store.get_trace_events(trace_id)
        new_result = [e for e in result if e.event_type == EventType.TOOL_CALL]
        assert len(new_result) == 5
        ts = [e.timestamp for e in new_result]
        assert ts == sorted(ts), "Events not ordered ASC by timestamp"
        print("[OK] get_trace_events ordering")

        # -- trace CRUD -------------------------------------------------------
        trace = Trace(agent_id="smoke-agent", start_time=t0)
        await trace_store.save_trace(trace)
        loaded_t = await trace_store.get_trace(trace.trace_id)
        assert loaded_t.agent_id == "smoke-agent"
        print("[OK] Trace save/get")

        await trace_store.update_trace(trace.trace_id, {"status": EventStatus.SUCCESS})
        updated = await trace_store.get_trace(trace.trace_id)
        assert updated.status == EventStatus.SUCCESS
        print("[OK] Trace update")

        # -- count_traces -----------------------------------------------------
        for i in range(5):
            t = Trace(
                agent_id="agent-A",
                start_time=t0 + timedelta(hours=i),
                status=EventStatus.PENDING,
            )
            await trace_store.save_trace(t)
        cnt = await trace_store.count_traces(agent_id="agent-A")
        assert cnt == 5, f"Expected 5, got {cnt}"
        print("[OK] count_traces")

        # -- list_trace_page keyset pagination --------------------------------
        all_pages: list[Trace] = []
        cursor = None
        iterations = 0
        while True:
            page, cursor = await trace_store.list_trace_page(
                limit=2, agent_id="agent-A", cursor=cursor
            )
            all_pages.extend(page)
            iterations += 1
            if cursor is None:
                break
            if iterations > 10:
                raise RuntimeError("pagination loop")
        assert len(all_pages) == 5, f"Expected 5 traces across pages, got {len(all_pages)}"
        starts = [t.start_time for t in all_pages]
        assert starts == sorted(starts, reverse=True), "Pages not in DESC order"
        print("[OK] list_trace_page keyset pagination")

        print("\nAll smoke tests passed.")

    finally:
        await db.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
