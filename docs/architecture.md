# AgentLens Architecture

**Adaptive Causal Observability for Cost-Efficient Debugging of Autonomous AI Agents**

## Research Problem

How can we minimize the cost and overhead of observing an AI agent while retaining enough causal information to diagnose its failures accurately and quickly?

Modern AI agents perform:
- LLM calls
- Tool calls
- RAG/retrieval
- Memory operations
- Planning/decision steps
- Retries
- Sub-agent executions
- API calls

When an agent fails, simply having a list of logs is often insufficient to understand **why** it failed.

## Core Trade-off

```
    OBSERVABILITY COST
            ↓
  CPU / Network / Storage
  Telemetry / Latency
            ↕
    DIAGNOSTIC QUALITY
            ↓
  Accuracy / Evidence / Speed
```

## High-Level Architecture

```
                ┌─────────────────────┐
                │      AI AGENT       │
                │                     │
                │ LLM / Tools / RAG   │
                │ Memory / Sub-agents │
                └──────────┬──────────┘
                           │
                     Agent Events
                           │
                           ▼
                ┌─────────────────────┐
                │    AgentLens SDK    │
                │                     │
                │ Event Collector     │
                │ Event Normalizer    │
                │ Context Manager     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Adaptive Observation│
                │ Engine              │
                │                     │
                │ Importance Scoring  │
                │ Observation Policy  │
                └──────────┬──────────┘
                           │
                ┌──────────┼──────────┐
                ▼          ▼          ▼
              FULL      COMPRESS     DROP
                │          │
                └─────┬────┘
                      ▼
            ┌────────────────────┐
            │  Causal Graph      │
            │  Builder           │
            └─────────┬──────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
         Event Store       Graph Store
              │                │
              └───────┬────────┘
                      ▼
            ┌────────────────────┐
            │ Failure Analyzer   │
            │                    │
            │ Anomaly Detection  │
            │ Root Cause Analysis│
            │ Evidence Retrieval │
            └─────────┬──────────┘
                      │
                      ▼
            ┌────────────────────┐
            │ AgentLens Dashboard│
            │                    │
            │ Timeline           │
            │ Causal Graph       │
            │ Root Cause         │
            │ Evidence           │
            │ Metrics            │
            └────────────────────┘
```

## Core Concepts

### A. Agent Event

Every important agent operation becomes an `AgentEvent`.

**Examples:**
- `LLM_CALL`
- `TOOL_CALL`
- `RETRIEVAL`
- `MEMORY_READ` / `MEMORY_WRITE`
- `SUBAGENT_START` / `SUBAGENT_END`
- `DECISION`
- `ERROR`
- `RETRY`
- `FINAL_RESPONSE`

**Event properties:**
- `event_id` — Unique identifier
- `trace_id` — Groups events from one execution
- `parent_id` — Hierarchical execution structure
- `agent_id` — Which agent created this
- `event_type` — Operation type (from enum)
- `timestamp` — When it occurred
- `input` — Input to operation
- `output` — Result
- `status` — Success/failure
- `latency` — Duration
- `metadata` — Contextual information
- `importance_score` — *(Phase 2)* How important for diagnosis?
- `observation_level` — *(Phase 2)* FULL / COMPRESSED / METADATA / DROP

### B. Trace

A trace represents one complete agent execution.

**Example:**

```
User Request
    ↓
Planner (LLM)
    ↓
Search Tool
    ↓
Retrieval
    ↓
LLM Decision
    ↓
Final Answer
```

All events share a `trace_id`.

### C. Parent-Child Relationships

Events support hierarchical execution.

**Example:**

```
Parent Agent
 ├── Planner (LLM)
 ├── Search Tool
 └── Research Sub-agent
       ├── LLM Call
       └── Search Tool
```

### D. Adaptive Observability

*Phase 2 feature:*

Events are classified into observation levels:

- **FULL** — Capture detailed information
- **COMPRESSED** — Store reduced information/summary
- **METADATA_ONLY** — Store minimal information
- **DROP** — Do not retain the event payload

Initial policy will be rule-based. No ML/RL in Phase 1.

### E. Causal Graph

*Phase 3 feature:*

Represent relationships between events:

- `PARENT_OF`
- `DEPENDS_ON`
- `INFLUENCED`
- `CAUSED`
- `RETRIEVED_FOR`
- `GENERATED_FROM`

**Example:**

```
Incorrect Retrieval
        │
        │ influenced
        ▼
LLM Decision
        │
        │ caused
        ▼
Wrong Final Answer
```

### F. Failure Diagnosis

*Phase 4 feature:*

When an agent fails, AgentLens answers: **"What caused the failure?"**

**Example:**

```
ROOT CAUSE:
  Incorrect result returned by Search Tool.

EVIDENCE PATH:
  Search Tool
      ↓
  Incorrect Search Result
      ↓
  LLM Decision
      ↓
  Final Answer
```

The diagnosis is grounded in recorded execution evidence—not an unsupported LLM explanation.

## Functional Requirements

| ID  | Requirement | Phase |
|-----|-------------|-------|
| FR1 | Agent Instrumentation | 1 |
| FR2 | Event Management | 1 |
| FR3 | Trace Management | 1 |
| FR4 | Hierarchical Context | 1 |
| FR5 | Event Storage | 1 |
| FR6 | Adaptive Observation | 2 |
| FR7 | Causal Graph | 3 |
| FR8 | Failure Detection | 4 |
| FR9 | Root Cause Analysis | 4 |
| FR10 | Evidence Retrieval | 4 |
| FR11 | Dashboard | 5 |
| FR12 | Benchmarking | 6 |
| FR13 | Metrics | 6 |

## Non-Functional Requirements

- **NFR1** — Low Overhead
- **NFR2** — Asynchronous Design
- **NFR3** — Scalability (hundreds/thousands of events per trace)
- **NFR4** — Reliability (fail-open)
- **NFR5** — Extensibility
- **NFR6** — Reproducibility
- **NFR7** — Testability
- **NFR8** — Privacy (configurable payload capture)
- **NFR9** — Maintainability (type hints, documentation)
- **NFR10** — Self-Observability (AgentLens metrics)

## Development Phases

### Phase 1 — Foundation ✓ (Current)
- AI Agent
- AgentLens SDK
- Event Collector
- Structured AgentEvent
- JSON/in-memory storage

### Phase 2 — Adaptive Observability
- Event importance scoring
- Observation policy (FULL / COMPRESSED / METADATA / DROP)

### Phase 3 — Causal Graph
- Event relationships
- Causal graph construction

### Phase 4 — Failure Diagnosis
- Failure detection
- Root cause analysis
- Evidence retrieval

### Phase 5 — Dashboard
- FastAPI backend
- React frontend
- Trace timeline
- Causal graph visualization

### Phase 6 — Research Benchmark
- Comparison: No tracing vs Full tracing vs AgentLens
- Metrics collection and analysis

## Research Questions

**Primary:** Can adaptive observability reduce telemetry and system overhead while preserving accurate and fast diagnosis of AI-agent failures?

**Supporting Questions:**

- RQ1: How does observability overhead change with operations per trace?
- RQ2: How much telemetry reduction before diagnostic accuracy drops?
- RQ3: Can causal execution graphs improve root-cause localization?
- RQ4: What is the trade-off between cost and diagnostic accuracy?

**Hypotheses (to be validated):**

- H1: Adaptive tracing reduces telemetry volume vs. full tracing
- H2: Adaptive tracing retains comparable failure-diagnosis accuracy
- H3: Causal graphs improve failure localization vs. flat logs
- H4: High-fanout/multi-agent workloads expose greater overhead

## Technology Stack

**Backend/Core:**
- Python 3.12+
- Pydantic
- FastAPI (Phase 5)
- asyncio

**Storage:**
- PostgreSQL (structured events) — Phase 1+
- Neo4j (causal graph) — Phase 3+

**Frontend:**
- React
- TypeScript

**Monitoring:**
- Prometheus
- Grafana (optional)

**Infrastructure:**
- Docker
- Docker Compose
- Linux

**Research:**
- pandas
- NumPy
- matplotlib

## Principles

1. Keep the framework independent from specific observability vendors
2. Prefer simple implementations before complex/ML-based solutions
3. Keep experiments reproducible
4. Separate core AgentLens from experimental/demo code
5. Use type hints
6. Write tests for important behavior
7. Avoid unnecessary abstractions
8. Don't implement future phases prematurely
9. Keep APIs modular for independent team work
10. Document major architectural decisions

## Repository Structure

```
AgentLens/
├── src/
│   └── agentlens/
│       ├── __init__.py
│       ├── events.py           # Event models
│       ├── collector.py        # Event collection interface
│       ├── normalizer.py       # (Phase 1+) Event normalization
│       ├── storage/            # (Phase 1+) Storage backends
│       ├── analysis/           # (Phase 4+) Failure diagnosis
│       └── dashboard/          # (Phase 5+) Web API & frontend
├── examples/
│   └── simple_agent.py         # Example agent
├── tests/
│   └── test_*.py               # Unit & integration tests
├── docs/
│   ├── architecture.md         # This file
│   └── phases.md               # Phase details
├── pyproject.toml              # Project metadata & dependencies
├── README.md                   # Quick start
└── .gitignore                  # Git ignore rules
```

---

**Project Status:** Phase 1 Repository Skeleton  
**Last Updated:** 2026-08-31
