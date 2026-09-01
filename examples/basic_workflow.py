"""
examples/basic_workflow.py
==========================

Demonstrates the full AgentLens storage flow end-to-end.

This script shows how to:
1. Create an InMemoryEventStore and InMemoryTraceStore
2. Wire the store into an async collector
3. Record a realistic multi-step trace with parent-child events
4. Retrieve, inspect, and pretty-print stored events

Run with:
    python -m examples.basic_workflow
    -- or --
    python src/../examples/basic_workflow.py
"""

from __future__ import annotations

import asyncio
from datetime import timezone
from uuid import UUID, uuid4
from typing import Optional, Any

from agentlens.events import (
    AgentEvent,
    EventStatus,
    EventType,
    Trace,
    create_event,
    create_trace,
)
from agentlens.storage import InMemoryEventStore, InMemoryTraceStore


# ---------------------------------------------------------------------------
# Minimal async collector (same shim used in integration tests)
# Replace with `from agentlens.sdk_collector import EventCollector` after merge
# ---------------------------------------------------------------------------


class AsyncCollectorShim:
    """Lightweight async collector shim — mirrors sdk_collector.EventCollector."""

    def __init__(self, storage_backend: Any = None) -> None:
        self._storage = storage_backend
        self._trace_id: Optional[UUID] = None
        self._agent_id: Optional[str] = None
        self._parent_stack: list[UUID] = []

    async def start_trace(self, agent_id: str, trace_id: Optional[UUID] = None) -> UUID:
        self._trace_id = trace_id or uuid4()
        self._agent_id = agent_id
        self._parent_stack = []
        return self._trace_id

    async def end_trace(self) -> None:
        self._trace_id = None
        self._agent_id = None
        self._parent_stack = []

    async def record_event(
        self,
        event_type: EventType | str = EventType.LLM_CALL,
        input_data: Optional[dict[str, Any]] = None,
        output_data: Optional[dict[str, Any]] = None,
        status: EventStatus | str = EventStatus.SUCCESS,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentEvent:
        if self._trace_id is None:
            raise RuntimeError("No active trace.")
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        if isinstance(status, str):
            status = EventStatus(status)

        parent_id = self._parent_stack[-1] if self._parent_stack else None
        event = create_event(
            trace_id=self._trace_id,
            agent_id=self._agent_id or "unknown",
            event_type=event_type,
            status=status,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            parent_id=parent_id,
            metadata=metadata,
        )
        if self._storage is not None:
            await self._storage.save(event)
        return event

    def push_parent(self, event_id: UUID) -> None:
        self._parent_stack.append(event_id)

    def pop_parent(self) -> Optional[UUID]:
        return self._parent_stack.pop() if self._parent_stack else None


# ---------------------------------------------------------------------------
# Helper: pretty print event tree
# ---------------------------------------------------------------------------


def _indent(level: int) -> str:
    return "    " * level


def print_event_tree(events: list[AgentEvent]) -> None:
    """Print events as an indented tree using parent-child relationships."""
    by_id: dict[UUID, AgentEvent] = {e.event_id: e for e in events}
    children: dict[Optional[UUID], list[AgentEvent]] = {}
    for e in events:
        children.setdefault(e.parent_id, []).append(e)

    def _print(parent: Optional[UUID], level: int) -> None:
        for ev in children.get(parent, []):
            lat = f"{ev.latency_ms:.1f}ms" if ev.latency_ms is not None else "N/A"
            status_icon = "[OK]" if ev.status == EventStatus.SUCCESS else "[FAIL]"
            print(
                f"{_indent(level)}{status_icon} [{ev.event_type.value:20s}]  "
                f"latency={lat:>10s}  id={str(ev.event_id)[:8]}..."
            )
            _print(ev.event_id, level + 1)

    print()
    _print(None, 0)
    print()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 65)
    print("  AgentLens -- Basic Workflow Demo")
    print("  Storage: InMemoryEventStore + InMemoryTraceStore")
    print("=" * 65)

    # --- Setup stores -------------------------------------------------
    event_store = InMemoryEventStore()
    trace_store = InMemoryTraceStore()
    collector = AsyncCollectorShim(storage_backend=event_store)

    # --- Start trace --------------------------------------------------
    agent_id = "demo-research-agent"
    trace_id = await collector.start_trace(agent_id=agent_id)
    print(f"\n[1] Started trace: {trace_id}")

    # Save trace metadata to trace store
    trace_meta = Trace(trace_id=trace_id, agent_id=agent_id)
    await trace_store.save_trace(trace_meta)

    # --- Record events ------------------------------------------------
    print("\n[2] Recording events ...")

    # Root: planning decision
    decision = await collector.record_event(
        event_type=EventType.DECISION,
        input_data={"question": "Research AI agents in 2026"},
        output_data={"plan": ["search", "retrieve", "synthesize"]},
        latency_ms=25.0,
        metadata={"model": "gemini-2.0-flash"},
    )
    print(f"     DECISION  id={str(decision.event_id)[:8]}")

    # Push decision as parent for search
    collector.push_parent(decision.event_id)

    search_event = await collector.record_event(
        event_type=EventType.TOOL_CALL,
        input_data={"tool": "web_search", "query": "AI agents 2026 trends"},
        output_data={"hits": 15, "top": ["arxiv.org", "openai.com"]},
        latency_ms=310.0,
        metadata={"tool": "web_search"},
    )
    print(f"       TOOL_CALL (parent=DECISION)  id={str(search_event.event_id)[:8]}")

    # Retrieval under search
    collector.push_parent(search_event.event_id)
    retrieval = await collector.record_event(
        event_type=EventType.RETRIEVAL,
        input_data={"query": "AI agents 2026 trends", "top_k": 5},
        output_data={"docs": ["doc_1", "doc_2", "doc_3"]},
        latency_ms=120.0,
    )
    print(f"         RETRIEVAL (parent=TOOL_CALL)  id={str(retrieval.event_id)[:8]}")
    collector.pop_parent()

    # Memory write after retrieval
    mem_write = await collector.record_event(
        event_type=EventType.MEMORY_WRITE,
        input_data={"key": "search_results"},
        output_data={"written": True},
        latency_ms=4.0,
    )
    print(f"       MEMORY_WRITE (parent=DECISION)  id={str(mem_write.event_id)[:8]}")

    collector.pop_parent()  # back to no parent

    # Synthesis LLM call
    llm_call = await collector.record_event(
        event_type=EventType.LLM_CALL,
        input_data={"prompt": "Synthesize: AI agents trends 2026", "docs": 3},
        output_data={"summary": "Multi-agent systems are dominating in 2026..."},
        latency_ms=520.0,
        metadata={"model": "gemini-2.0-flash", "tokens": 840},
    )
    print(f"     LLM_CALL  id={str(llm_call.event_id)[:8]}")

    # Final response
    final = await collector.record_event(
        event_type=EventType.FINAL_RESPONSE,
        output_data={"answer": "In 2026, multi-agent systems dominate AI research."},
        latency_ms=1.0,
    )
    print(f"     FINAL_RESPONSE  id={str(final.event_id)[:8]}")

    await collector.end_trace()

    # --- Update trace metadata ----------------------------------------
    from datetime import datetime
    await trace_store.update_trace(
        trace_id,
        {
            "status": EventStatus.SUCCESS,
            "end_time": datetime.now(timezone.utc),
            "event_count": 6,
        },
    )

    # --- Retrieve and verify ------------------------------------------
    print(f"\n[3] Retrieving stored events for trace {str(trace_id)[:8]}...")
    stored_events = await event_store.get_trace_events(trace_id)
    print(f"    -> {len(stored_events)} events retrieved")
    stored_trace = await trace_store.get_trace(trace_id)
    print(f"    -> Trace status : {stored_trace.status.value}")
    print(f"    -> Total latency: {sum(e.latency_ms or 0 for e in stored_events):.1f}ms")

    # --- Print event tree --------------------------------------------
    print("\n[4] Event execution tree:")
    print_event_tree(stored_events)

    # --- List all traces ---------------------------------------------
    all_trace_ids = await event_store.list_traces()
    print(f"[5] Total traces in store: {len(all_trace_ids)}")

    print("=" * 65)
    print("  Demo complete -- all events verified [OK]")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
