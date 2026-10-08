"""Run a multi-step AgentLens tracing demonstration.

The demo uses Gemini for planning and synthesis when GEMINI_API_KEY is set.
Without an API key it uses deterministic local responses, so the tracing and
storage demonstration remains runnable offline.

Run from the repository root:

    python -m examples.multistep_agent_demo
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from uuid import UUID

from agentlens.events import EventStatus, EventType, Trace
from agentlens.sdk_collector import EventCollector, EventNormalizer
from agentlens.storage import InMemoryEventStore, InMemoryTraceStore


class MultiStepResearchAgent:
    """Small research workflow instrumented with AgentLens events."""

    def __init__(self, event_store: InMemoryEventStore) -> None:
        self.collector = EventCollector(storage_backend=event_store)
        self.normalizer = EventNormalizer()
        self._gemini_client: Any = None

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=api_key)
            except ImportError:
                print("Gemini dependency unavailable; using offline responses.")

    async def _ask_gemini(self, prompt: str, fallback: str) -> str:
        """Ask Gemini synchronously through a worker thread, with a fallback."""
        if self._gemini_client is None:
            return fallback
        try:
            response = await asyncio.to_thread(
                self._gemini_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as exc:
            print(f"Gemini call unavailable ({exc}); continuing with offline response.")
            return fallback

    async def run(self, query: str) -> tuple[UUID, Trace]:
        trace_id = await self.collector.start_trace(agent_id="multistep-research-agent")
        started = time.perf_counter()

        try:
            planning_prompt = f"Create a concise three-step research plan for: {query}"
            print("Step 1/5: Asking Gemini to create a research plan.")
            print(f"Prompt: {planning_prompt}")
            plan = await self._ask_gemini(
                planning_prompt,
                "1. Search current sources 2. Retrieve useful evidence 3. Synthesize findings",
            )
            planning_event = await self.collector.record_event(
                event_type=EventType.DECISION,
                input_data={"query": query},
                output_data={"plan": plan},
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata={"step": "planning"},
            )

            self.collector.push_parent(planning_event.event_id)
            print("Step 2/5: Recording a web-search tool call.")
            search_started = time.perf_counter()
            search_event = await self.collector.record_event(
                event_type=EventType.TOOL_CALL,
                input_data={"tool": "web_search", "query": query},
                output_data={
                    "hits": ["research-paper-1", "research-paper-2", "technical-report-1"]
                },
                latency_ms=(time.perf_counter() - search_started) * 1000,
                metadata={"step": "search", "tool": "web_search"},
            )

            self.collector.push_parent(search_event.event_id)
            print("Step 3/5: Recording retrieved evidence under the search call.")
            retrieval_started = time.perf_counter()
            retrieval_event = await self.collector.record_event(
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
            self.collector.pop_parent()

            print("Step 4/5: Recording the evidence in agent memory.")
            await self.collector.record_event(
                event_type=EventType.MEMORY_WRITE,
                input_data={"key": "retrieved_evidence"},
                output_data={"document_count": len(retrieval_event.output_data or {})},
                latency_ms=2.0,
                metadata={"step": "memory-write"},
            )
            self.collector.pop_parent()

            synthesis_started = time.perf_counter()
            synthesis_prompt = (
                f"Answer this query using the plan and evidence below:\nQuery: {query}\n"
                f"Plan: {plan}\nEvidence: {retrieval_event.output_data}"
            )
            print("Step 5/5: Asking Gemini to synthesize the final answer.")
            print(f"Prompt: {synthesis_prompt}")
            answer = await self._ask_gemini(
                synthesis_prompt,
                "The evidence suggests a measurable, traceable workflow with planning, retrieval, and synthesis.",
            )
            await self.collector.record_event(
                event_type=EventType.LLM_CALL,
                input_data={"prompt": query, "plan": plan},
                output_data={"response": answer},
                latency_ms=(time.perf_counter() - synthesis_started) * 1000,
                metadata={"step": "synthesis", "model": "gemini-2.5-flash"},
            )

            await self.collector.record_event(
                event_type=EventType.FINAL_RESPONSE,
                output_data={"answer": answer},
                latency_ms=0.0,
                metadata={"step": "final-response"},
            )
            await self.collector.end_trace()
            print("Trace complete. Preparing final statistics.")

            stored_events = await self._event_store.get_trace_events(trace_id)
            trace = Trace(
                trace_id=trace_id,
                agent_id="multistep-research-agent",
                status=EventStatus.SUCCESS,
                events=stored_events,
                event_count=len(stored_events),
                total_latency_ms=sum(event.latency_ms or 0.0 for event in stored_events),
            )
            trace_meta = self._trace_store
            await trace_meta.save_trace(trace)
            await trace_meta.update_trace(
                trace_id,
                {"end_time": trace.end_time, "status": EventStatus.SUCCESS},
            )
            return trace_id, await trace_meta.get_trace(trace_id)
        except Exception:
            await self.collector.end_trace()
            raise

    async def execute(self, query: str) -> tuple[UUID, Trace]:
        """Run the workflow with stores available to every step."""
        self._event_store = self.collector._storage
        self._trace_store = InMemoryTraceStore()
        return await self.run(query)


async def main() -> None:
    query = (
        "How can observability reduce the cost of debugging autonomous AI agents "
        "while preserving enough information to diagnose failures?"
    )
    event_store = InMemoryEventStore()
    agent = MultiStepResearchAgent(event_store)
    trace_id, trace = await agent.execute(query)

    # Exercise the normalizer on a representative external-style event shape.
    normalized = agent.normalizer.from_dict(
        {"agent": "external-agent", "type": "tool_call", "duration_ms": 12.5}
    )
    event_types = [event.event_type.value for event in trace.events]

    print("\nAgentLens multi-step demo")
    print("=" * 64)
    print(f"Trace ID       : {trace_id}")
    print(f"Events stored  : {len(trace.events)}")
    print(f"Total latency  : {trace.total_latency_ms:.2f} ms")
    print(f"Normalized     : {normalized.event_type.value}, {normalized.latency_ms} ms")
    print("Event sequence : " + " -> ".join(event_types))
    print("\nExecution tree:")
    for event in trace.events:
        parent = str(event.parent_id)[:8] if event.parent_id else "root"
        print(
            f"  {event.event_type.value:16s} status={event.status.value:7s} "
            f"parent={parent:8s} latency={event.latency_ms or 0:.2f}ms"
        )
    print("\nDemo completed successfully: events were traced, normalized, and stored.")


if __name__ == "__main__":
    asyncio.run(main())
