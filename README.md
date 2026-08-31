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

**Status:** Phase 1 - Repository Skeleton  
**Last Updated:** 2026-08-31
