"""
AgentLens SDK Collector
=======================

High-level SDK for capturing and normalizing agent execution events.

This package provides:

- ``ContextManager``  — async-safe trace context tracking via ``contextvars``
- ``EventCollector``  — core SDK class for recording events and managing traces
- ``EventNormalizer`` — adapter that converts arbitrary dicts / framework events
                        into normalized ``AgentEvent`` instances

Quickstart::

    from agentlens.sdk_collector import EventCollector

    collector = EventCollector()

    async def my_agent():
        async with collector.traced(agent_id="my-agent"):
            event = await collector.record_event(
                event_type="llm_call",
                input_data={"prompt": "Hello"},
                output_data={"response": "World"},
                status="success",
                latency_ms=120.5,
                metadata={"model": "gemini-2.0-flash"},
            )
"""

from agentlens.sdk_collector.collector import EventCollector
from agentlens.sdk_collector.context import ContextManager
from agentlens.sdk_collector.normalizer import EventNormalizer

__all__ = [
    "ContextManager",
    "EventCollector",
    "EventNormalizer",
]
