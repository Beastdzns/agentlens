# AgentLens

**Adaptive Causal Observability for Cost-Efficient Debugging of Autonomous AI Agents**

A research project investigating how to minimize observability overhead while preserving accurate diagnosis of AI-agent failures.

## Quick Start

### Prerequisites

- Python 3.12 or later
- pip or uv

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/agentlens.git
   cd agentlens
   ```

2. **Create a virtual environment:**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the project in development mode:**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs:
   - Core dependencies: `pydantic`
   - Development tools: `pytest`, `pytest-asyncio`, `black`, `ruff`, `mypy`

### Running Tests

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src/agentlens --cov-report=html
```

### Code Quality

Format code:

```bash
black src/ tests/ examples/
```

Lint code:

```bash
ruff check src/ tests/ examples/
```

Type check:

```bash
mypy src/
```

## Project Structure

```
AgentLens/
├── src/agentlens/          # Core package
│   ├── __init__.py         # Package initialization
│   ├── events.py           # Event models & schemas
│   └── collector.py        # Event collection interface
├── examples/               # Example code & agents
│   └── simple_agent.py    # Simple test agent (placeholder)
├── tests/                  # Test suite
│   └── __init__.py
├── docs/
│   └── architecture.md     # Architecture & design
├── pyproject.toml          # Project metadata
├── README.md              # This file
└── .gitignore
```

## Architecture Overview

AgentLens captures structured execution events from AI agents, analyzes them for failure diagnosis, and visualizes causal relationships.

```
AI Agent
    ↓
Event Collector (SDK)
    ↓
Adaptive Observation Engine
    ↓
Causal Graph Builder
    ↓
Failure Analyzer
    ↓
Dashboard
```

See [docs/architecture.md](docs/architecture.md) for detailed design.

## Development Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation (events, collector, storage) | Current |
| 2 | Adaptive observability (importance scoring) | Planned |
| 3 | Causal graphs | Planned |
| 4 | Failure diagnosis & root cause | Planned |
| 5 | Web dashboard | Planned |
| 6 | Research benchmark | Planned |

## Research Questions

**Primary Question:**
Can adaptive observability reduce telemetry and system overhead while preserving accurate and fast diagnosis of AI-agent failures?

**Key Hypotheses:**
- H1: Adaptive tracing reduces telemetry volume vs. full tracing
- H2: Adaptive tracing preserves failure-diagnosis accuracy
- H3: Causal graphs improve root-cause localization
- H4: High-fanout workloads expose greater overhead

## Technology Stack

**Core:**
- Python 3.12+
- Pydantic (data validation)
- asyncio (async execution)

**Storage (future):**
- PostgreSQL (structured events)
- Neo4j (causal graphs)

**Frontend (future):**
- FastAPI
- React + TypeScript

**Testing:**
- pytest
- pytest-asyncio

## Design Principles

1. **Vendor-independent** — Not tied to Langfuse or other specific observability vendors
2. **Simple first** — Prefer straightforward solutions over premature optimization
3. **Reproducible** — Experiments use fixed configurations
4. **Modular** — Clear boundaries between components
5. **Typed** — Full type hints for maintainability
6. **Tested** — Important behavior covered by tests
7. **Documented** — Architecture and decisions clearly explained

## Contributing

This is an academic research project. Contributions welcome!

Please:
- Follow the existing code style (black, ruff)
- Add type hints
- Include tests for new features
- Document architectural changes
- Keep phases separate (don't implement Phase N+1 prematurely)

## License

MIT

## Citation

If you use AgentLens in your research, please cite:

```
@mastersthesis{agentlens2026,
  title={Adaptive Causal Observability for Cost-Efficient Debugging of Autonomous AI Agents},
  author={Your Name},
  school={Your University},
  year={2026}
}
```

## Getting Help

- **Architecture:** See [docs/architecture.md](docs/architecture.md)
- **Issues:** File a GitHub issue
- **Discussions:** GitHub Discussions

---

**Status:** Phase 1 - Repository Skeleton → SDK Collector Added
**Last Updated:** 2026-09-01

---

## SDK Collector Module (`sdk_collector/`)

> Added in Phase 1 by: [your name here]

The `sdk_collector` package (`src/agentlens/sdk_collector/`) provides the
high-level SDK interface for capturing and normalizing agent events.  It is
built on top of Shridhar's `AgentEvent` Pydantic models and extends them with
fully async-safe, context-propagating trace management.

### Components

| Class | File | Responsibility |
|---|---|---|
| `ContextManager` | `context.py` | `contextvars`-backed trace/parent ID storage; task-isolated |
| `EventCollector` | `collector.py` | Core SDK — `record_event`, `start_trace`, `end_trace`, `@traced` |
| `EventNormalizer` | `normalizer.py` | Converts arbitrary dicts / framework events to `AgentEvent` |

### Key Capabilities

**`ContextManager`**

```python
from agentlens.sdk_collector import ContextManager
from uuid import uuid4

ctx = ContextManager()
ctx.set_trace_id(uuid4())
ctx.push_parent(some_event_id)   # nest events
print(ctx.get_context())         # {"trace_id": ..., "parent_id": ..., "agent_id": ...}
ctx.pop_parent()
```

**`EventCollector` — manual lifecycle**

```python
from agentlens.sdk_collector import EventCollector
from agentlens.events import EventType

collector = EventCollector()          # pass storage_backend= to persist

await collector.start_trace(agent_id="my-agent")

event = await collector.record_event(
    event_type=EventType.LLM_CALL,
    input_data={"prompt": "Hello"},
    output_data={"response": "World"},
    status="success",
    latency_ms=110.0,
    metadata={"model": "gemini-2.0-flash"},
)

print(event.trace_id)   # auto-populated from context
await collector.end_trace()
```

**`EventCollector` — `@traced` decorator**

```python
@collector.traced(agent_id="search-agent")
async def run_search(query: str) -> str:
    await collector.record_event(
        event_type=EventType.TOOL_CALL,
        input_data={"query": query},
    )
    return "results"

# Works on regular def too:
@collector.traced(agent_id="sync-agent")
def compute(x: int) -> int:
    return x * 2
```

**`EventNormalizer` — dict conversion**

```python
from agentlens.sdk_collector import EventNormalizer

normalizer = EventNormalizer()
event = normalizer.from_dict({
    "agent_id": "research-bot",
    "event_type": "tool_call",
    "prompt": "What is AI?",          # aliased → input_data
    "response": {"answer": "..."},    # aliased → output_data
    "timestamp": "2026-01-01T12:00:00Z",
    "latency_ms": 88.5,
})
```

### Running the New Tests

```bash
# Activate your conda environment first
conda activate agentlens-env

# Run only the sdk_collector test suite
pytest tests/test_collector.py -v

# Run all tests (regression + new)
pytest -v

# Run with coverage report
pytest --cov=src/agentlens --cov-report=term-missing
pytest --cov=src/agentlens --cov-report=html   # open htmlcov/index.html
```

### Conda Environment Setup

```bash
# Create a Python 3.12 environment
conda create -n agentlens-env python=3.12 -y

# Activate (Windows PowerShell / CMD)
conda activate agentlens-env

# Install the project in editable mode (includes all dev dependencies)
pip install -e ".[dev]"
```

---

## What to Change / Next Steps

### 🔧 Vyankatesh — Storage Backend Interface (Required)

The `EventCollector` accepts an optional `storage_backend` that is injected at
construction time.  **This interface is not yet defined.**  Vyankatesh needs to:

1. **Define the protocol** — create a class (or `typing.Protocol`) that exposes:
   ```python
   class StorageBackend(Protocol):
       async def save(self, event: AgentEvent) -> None: ...
   ```

2. **Implement concrete backends**, for example:
   - `InMemoryBackend` (for testing / local dev)
   - `PostgresBackend` (Phase 2 — structured event storage)
   - `Neo4jBackend` (Phase 3 — causal graph storage)

3. **Inject the backend**:
   ```python
   from my_storage import PostgresBackend

   backend = PostgresBackend(dsn="postgresql://...")
   collector = EventCollector(storage_backend=backend)
   ```

The `EventCollector` already calls `await storage_backend.save(event)` in a
**fail-open** try/except block, so any `StorageBackend` implementation simply
needs to implement `async save(event: AgentEvent) -> None`.

### Other Planned Work

- **LangChain adapter** — `EventNormalizer.from_langchain_event()` is a stub.
- **LlamaIndex adapter** — `EventNormalizer.from_llamaindex_event()` is a stub.
- **Importance scoring** — integrate with the `ObservationLevel` adaptive engine.
- **Causal graph builder** — Phase 3: build Neo4j graph from parent/child event links.

