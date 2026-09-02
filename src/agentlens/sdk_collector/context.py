"""
Async-safe trace context management for AgentLens SDK.

Uses :mod:`contextvars` so that every ``asyncio`` Task (coroutine) inherits an
independent *copy* of the context at creation time.  Mutations in one task do
**not** bleed into sibling or parent tasks — a fundamental requirement for
concurrent agent tracing.

Design notes
------------
- ``_parent_stack_var`` stores a *tuple* of UUIDs (immutable).  Each push
  creates a new tuple and writes it back via ``_parent_stack_var.set()``.
  This is copy-on-write and safe for concurrent task trees.
- All public methods are synchronous because ``ContextVar`` operations are
  themselves synchronous and thread-/async-safe.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional
from uuid import UUID


class ContextManager:
    """
    Manage the current trace context (trace ID, parent ID, agent ID)
    using :class:`contextvars.ContextVar`.

    Each ``asyncio`` Task inherits a snapshot of the context at spawn time,
    so concurrent traces remain completely isolated without any locking.

    Example::

        ctx = ContextManager()
        ctx.set_trace_id(uuid4())
        ctx.push_parent(some_event_id)
        print(ctx.get_context())
        ctx.pop_parent()
    """

    # Module-level ContextVars — shared across all instances so context
    # propagates naturally across the asyncio task tree.
    _trace_id_var: ContextVar[Optional[UUID]] = ContextVar(
        "agentlens_trace_id", default=None
    )
    _agent_id_var: ContextVar[Optional[str]] = ContextVar(
        "agentlens_agent_id", default=None
    )
    # Immutable tuple used as an append-only stack; copy-on-write semantics.
    _parent_stack_var: ContextVar[tuple[UUID, ...]] = ContextVar(
        "agentlens_parent_stack", default=()
    )

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_trace_id(self, trace_id: UUID) -> None:
        """
        Set the active trace ID for the current async context.

        Args:
            trace_id: UUID identifying the trace to activate.
        """
        self._trace_id_var.set(trace_id)

    def set_agent_id(self, agent_id: str) -> None:
        """
        Set the active agent ID for the current async context.

        Args:
            agent_id: String identifier for the agent.
        """
        self._agent_id_var.set(agent_id)

    def clear(self) -> None:
        """
        Reset all context variables to their defaults.

        Call this at the end of a trace to prevent context leaking into
        subsequent work scheduled on the same task.
        """
        self._trace_id_var.set(None)
        self._agent_id_var.set(None)
        self._parent_stack_var.set(())

    # ------------------------------------------------------------------
    # Parent stack operations
    # ------------------------------------------------------------------

    def push_parent(self, event_id: UUID) -> None:
        """
        Push *event_id* onto the parent stack, making it the current parent
        for any events recorded while it is on the stack.

        A new tuple is written back to the ContextVar on every call, so child
        tasks that were spawned *before* this push are unaffected.

        Args:
            event_id: The event ID that will act as the parent.
        """
        current = self._parent_stack_var.get()
        self._parent_stack_var.set(current + (event_id,))

    def pop_parent(self) -> Optional[UUID]:
        """
        Pop and return the most recently pushed parent event ID.

        Returns:
            The popped parent UUID, or ``None`` if the stack was empty.
        """
        current = self._parent_stack_var.get()
        if not current:
            return None
        popped = current[-1]
        self._parent_stack_var.set(current[:-1])
        return popped

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_trace_id(self) -> Optional[UUID]:
        """Return the active trace ID, or ``None`` if no trace is active."""
        return self._trace_id_var.get()

    def get_agent_id(self) -> Optional[str]:
        """Return the active agent ID, or ``None`` if not set."""
        return self._agent_id_var.get()

    def get_parent_id(self) -> Optional[UUID]:
        """
        Return the top of the parent stack (current parent event ID).

        Returns:
            The current parent UUID, or ``None`` if the stack is empty.
        """
        stack = self._parent_stack_var.get()
        return stack[-1] if stack else None

    def get_context(self) -> dict[str, Any]:
        """
        Return a snapshot of the current trace context as a plain dict.

        Returns:
            Dictionary with keys ``trace_id``, ``parent_id``, ``agent_id``.
            Values are ``None`` when not set.

        Example::

            {
                "trace_id": UUID("..."),
                "parent_id": None,
                "agent_id": "my-agent",
            }
        """
        return {
            "trace_id": self.get_trace_id(),
            "parent_id": self.get_parent_id(),
            "agent_id": self.get_agent_id(),
        }
