"""
Core SDK class for recording agent events and managing trace lifecycle.

Design Principles
-----------------
1. **Async-safe** — all mutable state lives in :class:`~contextvars.ContextVar`
   (via :class:`ContextManager`), not in instance variables, so multiple
   concurrent coroutines each see their own trace.

2. **Fail-open** — telemetry must never crash the agent.  Any exception raised
   by the optional ``storage_backend`` is caught, a warning is emitted, and
   execution continues normally.

3. **Injectable storage** — the ``storage_backend`` is anything that exposes an
   async ``save(event: AgentEvent) -> None`` method.  Pass ``None`` to run
   in-memory only (the event is still validated and returned).

4. **Decorator ergonomics** — ``@collector.traced(agent_id="…")`` works on
   both ``async def`` and regular ``def`` functions without the caller having
   to choose a different decorator.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import warnings
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from agentlens.events import AgentEvent, EventStatus, EventType, create_event
from agentlens.sdk_collector.context import ContextManager


class _TracedContext:
    """
    Returned by :meth:`EventCollector.traced`.  Acts simultaneously as:

    * An **async context manager** (``async with collector.traced(...)``).
    * A **decorator** for both ``async def`` and regular ``def`` functions.

    Instances are lightweight; create a new one via ``collector.traced()``.
    """

    def __init__(self, collector: "EventCollector", agent_id: str) -> None:
        self._collector = collector
        self._agent_id = agent_id

    # --- Async context manager protocol -----------------------------------

    async def __aenter__(self) -> "EventCollector":
        await self._collector.start_trace(agent_id=self._agent_id)
        return self._collector

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self._collector.end_trace()
        return False  # never suppress exceptions

    # --- Decorator protocol -----------------------------------------------

    def __call__(self, func: Callable) -> Callable:
        """Decorate *func*, wrapping it in the trace lifecycle."""
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self:
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Run a sync function inside a managed async trace context."""
                async def _run() -> Any:
                    async with self:
                        return func(*args, **kwargs)

                # If already inside an event loop (e.g. pytest-asyncio),
                # schedule as a task; otherwise bootstrap a new event loop.
                try:
                    loop = asyncio.get_running_loop()
                    return loop.create_task(_run())
                except RuntimeError:
                    return asyncio.run(_run())
            return sync_wrapper



class EventCollector:
    """
    Primary SDK interface for capturing structured agent execution events.

    Each :class:`EventCollector` instance shares the module-level
    :class:`ContextManager` (which uses ``contextvars``) so context always
    propagates correctly through ``asyncio`` task trees.

    Args:
        storage_backend: Optional object with an ``async save(event) -> None``
            method.  If *None*, events are validated and returned but not
            persisted.

    Example — manual trace lifecycle::

        collector = EventCollector()

        await collector.start_trace(agent_id="my-agent")
        event = await collector.record_event(
            event_type=EventType.LLM_CALL,
            input_data={"prompt": "Hello"},
            output_data={"response": "World"},
            status=EventStatus.SUCCESS,
            latency_ms=110.0,
        )
        await collector.end_trace()

    Example — decorator shorthand::

        @collector.traced(agent_id="my-agent")
        async def run_pipeline(query: str) -> str:
            await collector.record_event(...)
            return "done"
    """

    # Shared module-level context manager (ContextVar-backed, task-isolated).
    _ctx: ContextManager = ContextManager()

    def __init__(self, storage_backend: Any = None) -> None:
        """
        Initialise the collector.

        Args:
            storage_backend: Optional async storage interface.  Must expose
                ``async save(event: AgentEvent) -> None``.  Injected rather
                than created here so the collector stays testable.
        """
        self._storage = storage_backend

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------

    async def start_trace(
        self,
        agent_id: str,
        trace_id: Optional[UUID] = None,
    ) -> UUID:
        """
        Begin a new trace in the current async context.

        Sets the ``trace_id`` and ``agent_id`` ContextVars so all subsequent
        :meth:`record_event` calls within the same task (or child tasks) pick
        them up automatically.

        Args:
            agent_id: Identifier for the agent being traced.
            trace_id: Optional explicit trace UUID.  A new UUID is generated
                when omitted.

        Returns:
            The active ``trace_id`` UUID.
        """
        resolved_id = trace_id if trace_id is not None else uuid4()
        self._ctx.set_trace_id(resolved_id)
        self._ctx.set_agent_id(agent_id)
        return resolved_id

    async def end_trace(self) -> None:
        """
        Close the current trace and clear all context variables.

        Safe to call even if no trace was started (no-op in that case).
        """
        self._ctx.clear()

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    async def record_event(
        self,
        event_type: EventType | str,
        input_data: Optional[dict[str, Any]] = None,
        output_data: Optional[dict[str, Any]] = None,
        status: EventStatus | str = EventStatus.SUCCESS,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentEvent:
        """
        Record a single agent event, auto-populating IDs from the active context.

        The event is validated against :class:`~agentlens.events.AgentEvent`
        (Pydantic).  If a ``storage_backend`` is configured, the event is
        persisted asynchronously; any storage error is swallowed (fail-open).

        Args:
            event_type: One of :class:`~agentlens.events.EventType` (or its
                string value, e.g. ``"llm_call"``).
            input_data: Arbitrary dict representing the operation's input.
            output_data: Arbitrary dict representing the operation's output.
            status: Completion status (default ``SUCCESS``).
            latency_ms: Duration in milliseconds (must be ≥ 0 if provided).
            metadata: Additional key/value context attached to the event.

        Returns:
            The validated :class:`~agentlens.events.AgentEvent` instance.

        Raises:
            RuntimeError: If no trace is active (``start_trace`` not called).
        """
        ctx = self._ctx.get_context()
        trace_id: Optional[UUID] = ctx["trace_id"]
        agent_id: Optional[str] = ctx["agent_id"]
        parent_id: Optional[UUID] = ctx["parent_id"]

        if trace_id is None:
            raise RuntimeError(
                "No active trace.  Call `await collector.start_trace(agent_id)` "
                "or use the `@collector.traced(agent_id=...)` decorator."
            )

        # Coerce string event_type / status to their enum counterparts.
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        if isinstance(status, str):
            status = EventStatus(status)

        event = create_event(
            trace_id=trace_id,
            agent_id=agent_id or "unknown",
            event_type=event_type,
            status=status,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            parent_id=parent_id,
            metadata=metadata,
        )

        # Persist (fail-open).
        if self._storage is not None:
            try:
                await self._storage.save(event)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"AgentLens: storage backend failed to save event "
                    f"({event.event_id}): {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return event

    # ------------------------------------------------------------------
    # Context accessors
    # ------------------------------------------------------------------

    def get_current_trace_id(self) -> Optional[UUID]:
        """
        Return the active trace UUID, or ``None`` if no trace is running.
        """
        return self._ctx.get_trace_id()

    def get_current_parent_id(self) -> Optional[UUID]:
        """
        Return the current parent event UUID (top of the parent stack),
        or ``None`` if the stack is empty.
        """
        return self._ctx.get_parent_id()

    # ------------------------------------------------------------------
    # @traced async-context-manager decorator
    # ------------------------------------------------------------------

    def traced(self, agent_id: str) -> "_TracedContext":
        """
        Decorator / async context-manager that auto-manages the trace lifecycle.

        Works as a **decorator** on both ``async def`` and regular ``def``
        functions, and can also be used directly as an ``async with`` block.

        The decorator:

        1. Calls :meth:`start_trace` before the function body runs.
        2. Calls :meth:`end_trace` after the function body completes
           (including on exception).

        Args:
            agent_id: The agent identifier to associate with this trace.

        Returns:
            A :class:`_TracedContext` instance that is both a decorator and
            an async context manager.

        Example — decorator on ``async def``::

            @collector.traced(agent_id="search-agent")
            async def search(query: str) -> str:
                await collector.record_event(
                    event_type=EventType.TOOL_CALL,
                    input_data={"query": query},
                )
                return "results"

        Example — decorator on regular ``def``::

            @collector.traced(agent_id="sync-agent")
            def compute(x: int) -> int:
                return x * 2   # trace lifecycle managed automatically

        Example — async context manager::

            async with collector.traced(agent_id="pipeline"):
                await collector.record_event(...)
        """
        return _TracedContext(collector=self, agent_id=agent_id)
