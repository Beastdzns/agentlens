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

5. **Automatic failure capture** — ``capture_step()``, ``capture_llm_call()``,
   and ``capture_tool_call()`` are async context managers that time each step,
   record ``SUCCESS`` on clean exit, and record a ``FAILURE`` event (carrying
   ``exception_type`` + ``exception_message``) before re-raising on error.
   Retry linkage is supported via ``retry_of`` / ``attempt`` metadata.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import inspect
import time
import warnings
from typing import Any, AsyncIterator, Callable, Optional
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

    def __init__(
        self,
        storage_backend: Any = None,
        *,
        scorer: Any = None,
        policy: Any = None,
    ) -> None:
        """
        Initialise the collector.

        Args:
            storage_backend: Optional async storage interface.  Must expose
                ``async save(event: AgentEvent) -> None``.  Injected rather
                than created here so the collector stays testable.
            scorer: Optional :class:`~agentlens.observation.ImportanceScorer`.
                When supplied (together with *policy*), each event is scored
                and transformed before storage.
            policy: Optional :class:`~agentlens.observation.ObservationPolicy`.
                Paired with *scorer* to route events to an observation level.
        """
        self._storage = storage_backend
        self._scorer = scorer
        self._policy = policy
        # Per-trace repeat counters: trace_id (str) -> {event_type -> count}
        # defaultdict keeps this async-safe without explicit locking because
        # asyncio is single-threaded within an event loop.
        self._repeat_counters: dict[str, collections.Counter] = {}

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
        # Initialise a fresh repeat counter for this trace.
        self._repeat_counters[str(resolved_id)] = collections.Counter()
        return resolved_id

    async def end_trace(self) -> None:
        """
        Close the current trace and clear all context variables.

        Safe to call even if no trace was started (no-op in that case).
        """
        # Clean up the repeat counter for this trace.
        tid = self._ctx.get_trace_id()
        if tid is not None:
            self._repeat_counters.pop(str(tid), None)
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

        # --- Observation policy (D7: fail-open) ---------------------------------
        if self._scorer is not None and self._policy is not None:
            try:
                tid_str = str(trace_id)
                counter = self._repeat_counters.setdefault(tid_str, collections.Counter())
                repeat_idx = counter[event_type]
                counter[event_type] += 1

                score = self._scorer.score(event, repeat_index=repeat_idx)
                event = event.model_copy(update={"importance_score": score})

                level = self._policy.decide(score, event)
                event = self._policy.transform(event, level)
            except Exception as obs_exc:  # noqa: BLE001
                warnings.warn(
                    f"AgentLens: observation policy failed for event "
                    f"({event.event_id}): {obs_exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        # ------------------------------------------------------------------------

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

    def push_parent(self, event_id: UUID) -> None:
        """Make an event the parent of subsequently recorded events."""
        self._ctx.push_parent(event_id)

    def pop_parent(self) -> Optional[UUID]:
        """Remove and return the current parent event, if one exists."""
        return self._ctx.pop_parent()

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

    # ------------------------------------------------------------------
    # Automatic failure-capture context managers
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def capture_step(
        self,
        event_type: EventType | str,
        input_data: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        *,
        retry_of: Optional[UUID] = None,
        attempt: int = 1,
    ) -> AsyncIterator[None]:
        """
        Generic async context manager that captures a single pipeline step.

        Times execution of the body.  On clean exit it records a ``SUCCESS``
        event; on any exception it records a ``FAILURE`` event that carries
        ``exception_type`` and ``exception_message`` in *metadata*, then
        **re-raises** the original exception (traceback preserved).

        Retry linkage: if this attempt is a retry of a previous failed event,
        pass ``retry_of=<failed_event_id>`` and an incremented ``attempt``
        counter; both are stored in the event's metadata.

        Args:
            event_type: The :class:`~agentlens.events.EventType` for this step.
            input_data: Inputs to the step (recorded on both success and failure).
            metadata: Additional context merged into the recorded event's metadata.
            retry_of: UUID of the failed event this attempt is retrying.
            attempt: Attempt counter (1 = first try, 2 = first retry, …).

        Yields:
            Nothing — the body runs inside the ``async with`` block.

        Raises:
            Any exception raised inside the body (re-raised after recording).

        Example::

            async with collector.capture_step(
                EventType.LLM_CALL,
                input_data={"prompt": "..."},
                metadata={"step": "synthesis"},
            ):
                result = await call_llm(prompt)
        """
        _start = time.perf_counter()
        _base_meta: dict[str, Any] = dict(metadata or {})
        if retry_of is not None:
            _base_meta["retry_of"] = str(retry_of)
        if attempt > 1 or retry_of is not None:
            _base_meta["attempt"] = attempt

        try:
            yield
        except Exception as exc:
            _latency = (time.perf_counter() - _start) * 1000
            _fail_meta = {
                **_base_meta,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            await self.record_event(
                event_type=event_type,
                status=EventStatus.FAILURE,
                input_data=input_data,
                output_data=None,
                latency_ms=_latency,
                metadata=_fail_meta,
            )
            raise
        else:
            _latency = (time.perf_counter() - _start) * 1000
            await self.record_event(
                event_type=event_type,
                status=EventStatus.SUCCESS,
                input_data=input_data,
                latency_ms=_latency,
                metadata=_base_meta if _base_meta else None,
            )

    @contextlib.asynccontextmanager
    async def capture_llm_call(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash",
        metadata: Optional[dict[str, Any]] = None,
        *,
        retry_of: Optional[UUID] = None,
        attempt: int = 1,
    ) -> AsyncIterator[list[str]]:
        """
        Async context manager for an LLM call step.

        Yields a **result list** (a single-element mutable list) that the body
        should populate with the model response string so it can be recorded.
        Times execution, records ``SUCCESS`` (with prompt + response) on exit,
        or ``FAILURE`` (with exception details) if an exception escapes.

        Args:
            prompt: The input prompt sent to the model.
            model: Model identifier (default ``"gemini-2.0-flash"``).
            metadata: Extra metadata merged into the recorded event.
            retry_of: UUID of the failed event this call is retrying.
            attempt: Attempt counter.

        Yields:
            A ``list[str]`` with one empty-string element.  Assign the
            response to ``result[0]`` inside the ``async with`` body.

        Raises:
            Any exception raised inside the body (re-raised after recording).

        Example::

            async with collector.capture_llm_call(
                prompt="What is X?", model="gemini-2.5-flash"
            ) as result:
                result[0] = await call_llm(prompt)
            answer = result[0]
        """
        _start = time.perf_counter()
        result: list[str] = [""]
        _base_meta: dict[str, Any] = {"model": model, **(metadata or {})}
        if retry_of is not None:
            _base_meta["retry_of"] = str(retry_of)
        if attempt > 1 or retry_of is not None:
            _base_meta["attempt"] = attempt

        try:
            yield result
        except Exception as exc:
            _latency = (time.perf_counter() - _start) * 1000
            _fail_meta = {
                **_base_meta,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            await self.record_event(
                event_type=EventType.LLM_CALL,
                status=EventStatus.FAILURE,
                input_data={"prompt": prompt, "model": model},
                output_data=None,
                latency_ms=_latency,
                metadata=_fail_meta,
            )
            raise
        else:
            _latency = (time.perf_counter() - _start) * 1000
            await self.record_event(
                event_type=EventType.LLM_CALL,
                status=EventStatus.SUCCESS,
                input_data={"prompt": prompt, "model": model},
                output_data={"response": result[0]},
                latency_ms=_latency,
                metadata=_base_meta,
            )

    @contextlib.asynccontextmanager
    async def capture_tool_call(
        self,
        tool_name: str,
        input_data: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        *,
        retry_of: Optional[UUID] = None,
        attempt: int = 1,
    ) -> AsyncIterator[list[Any]]:
        """
        Async context manager for a tool-call step.

        Yields a **result list** (a single-element mutable list) that the body
        should populate with the tool's output dict so it can be recorded.
        Times execution, records ``SUCCESS`` (with input + output) on exit,
        or ``FAILURE`` (with exception details) if an exception escapes.

        Args:
            tool_name: Human-readable tool name stored in metadata.
            input_data: Tool inputs.
            metadata: Extra metadata merged into the recorded event.
            retry_of: UUID of the failed event this call is retrying.
            attempt: Attempt counter.

        Yields:
            A ``list`` with one ``None`` element.  Assign the output dict
            to ``result[0]`` inside the ``async with`` body.

        Raises:
            Any exception raised inside the body (re-raised after recording).

        Example::

            async with collector.capture_tool_call(
                "web_search", input_data={"query": q}
            ) as result:
                result[0] = await web_search(q)
            hits = result[0]
        """
        _start = time.perf_counter()
        result: list[Any] = [None]
        _base_meta: dict[str, Any] = {"tool": tool_name, **(metadata or {})}
        if retry_of is not None:
            _base_meta["retry_of"] = str(retry_of)
        if attempt > 1 or retry_of is not None:
            _base_meta["attempt"] = attempt

        try:
            yield result
        except Exception as exc:
            _latency = (time.perf_counter() - _start) * 1000
            _fail_meta = {
                **_base_meta,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            await self.record_event(
                event_type=EventType.TOOL_CALL,
                status=EventStatus.FAILURE,
                input_data=input_data,
                output_data=None,
                latency_ms=_latency,
                metadata=_fail_meta,
            )
            raise
        else:
            _latency = (time.perf_counter() - _start) * 1000
            await self.record_event(
                event_type=EventType.TOOL_CALL,
                status=EventStatus.SUCCESS,
                input_data=input_data,
                output_data=result[0] if isinstance(result[0], dict) else {"output": result[0]},
                latency_ms=_latency,
                metadata=_base_meta,
            )
