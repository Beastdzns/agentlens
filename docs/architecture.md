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

*Phase 10 feature (complete):*

Events are classified into observation levels:

- **FULL** — Capture detailed information
- **COMPRESSED** — Store reduced information/summary
- **METADATA_ONLY** — Store minimal information
- **DROP** — Do not retain the event payload

The policy is **rule-based** and **deterministic** (no ML/RL).

**Importance scoring** (`agentlens.observation.importance`) assigns each event a
score in `[0, 1]`:

```
raw = (base_by_status + weight_by_type) / 2 + modifiers   # then clamp to [0, 1]
```

| Contribution | Value |
|---|---|
| `base_by_status` | FAILURE `0.90`, PARTIAL `0.70`, PENDING `0.60`, SUCCESS `0.30` |
| `weight_by_type` | ERROR `1.00`, FINAL_RESPONSE `0.95`, RETRY `0.85`, DECISION `0.80`, SUBAGENT_* `0.70`, TOOL_CALL `0.55`, LLM_CALL/RETRIEVAL `0.50`, MEMORY_* `0.35` |
| `parent_bonus` | `+0.10` when `parent_id is None` (structural root) |
| `latency_bonus` | `+0.10` when `latency_ms >= 1000` |
| `exception_bonus` | `+0.10` when `metadata["exception_type"]` is set |
| `repetition_penalty` | `−0.15 × min(repeat_index, 3)` for repeated SUCCESS events (FINAL_RESPONSE / DECISION exempt) |

The two base contributions are **averaged, not summed**, so each stays an
independent `0..1` axis and the score lines up with the policy thresholds
(`0.60` FULL / `0.35` COMPRESSED / `0.15` METADATA_ONLY).

Errors, failures, and retries therefore score highest and are never repetition
penalised. The policy additionally forces `FAILURE`, `ERROR`, `RETRY` and
`FINAL_RESPONSE` to **FULL** so diagnostic evidence is never degraded, and
dropped events keep their identity and parent links so the causal graph stays
intact. All thresholds are configurable via `ScoringConfig` / `PolicyConfig` and
the `AGENTLENS_*` environment variables. Scoring and observation are **inactive
by default** — the collector only applies them when explicitly injected.

See `docs/phase10-adaptive-observability.md` for the evaluation results.

### E. Causal Graph

*Phase 11 feature:*

The causal graph makes the *structure* of a trace explicit, so a failure can be
read as a chain of causes instead of a flat list of events. It is built by
`agentlens.causal.graph.build_graph()` — a **pure** function over a list of
events: no I/O, no mutation of its input, and the same events always produce
the same graph.

**Node** — one event, carrying its identity, status, timings, observation
fields, a short label and (when recorded) the error message and exception type.

**Edges** — a **closed** taxonomy of four relations:

| Relation | Direction | Derived from | Traversed? |
|---|---|---|---|
| `parent_of` | parent → child | `event.parent_id` | yes |
| `retry_of` | failed attempt → retry | `metadata["retry_of"]` | yes |
| `depends_on` | node → dependency | `metadata["depends_on"]` | yes |
| `influenced` | failed node → child | a failing parent (annotation) | no |

`caused` is deliberately **not** a construction edge: it is the root-cause
analyzer's *verdict* (see §F), never one of its inputs. `influenced`
duplicates the `parent_of` direction and exists only so the UI can draw
failure propagation differently — it never creates ancestry.

**Guarantees:**

- **Deterministic** — nodes are ordered by `(timestamp, event_id)`, edges by
  `(relation, source, target)`; the input order does not matter.
- **Grounded** — every edge is derived from recorded fields. A parent id or a
  `retry_of` id outside the trace is ignored, never invented as a phantom node.
- **Graceful with missing evidence** — a dropped event keeps its node (with its
  `observation_level`) and its parent link, so the structure survives adaptive
  observation even when the payload does not.
- **Cycle-safe** — all traversal is iterative and visited-set guarded.

**Traversal API** (the substrate the analyzer builds on):

| Method | Meaning |
|---|---|
| `children(id)` / `parents(id)` | direct structural neighbours |
| `ancestors(id)` / `descendants(id)` | everything reachable upstream / downstream |
| `depth(id)` | length of the parent chain to a root (roots are `0`) |
| `path(src, dst)` | shortest structural path, or `None` |
| `failures()` | ids of every failed node |

**HTTP surface:**

- `GET /api/traces/{trace_id}/causal-graph` → `{trace_id, nodes[], edges[]}`
- `GET /api/traces/{trace_id}` → additive `causal_graph` field
  (opt out with `?include_causal_graph=false`)

Graph construction never turns a trace request into a 5xx: if a graph cannot be
built the field is omitted and the failure is logged (fail-open).

**Example:**

```
Incorrect Retrieval                 (failed)
        │
        │ parent_of / influenced
        ▼
LLM Decision
        │
        │ parent_of
        ▼
Wrong Final Answer
```

### F. Failure Diagnosis

*Phase 11 feature:*

When an agent fails, AgentLens answers: **"What caused the failure?"**

The analyzer (`agentlens.causal.analyzer.RootCauseAnalyzer`) walks the causal
graph with **rules only** — no model is ever called, so the same trace always
yields the same diagnosis (D1, NFR6).

**Diagnosis rules**

| # | Rule |
|---|---|
| 1 | A **failure** is a node with status `failure` (an `error`-typed node counts too) |
| 2 | A failure with **no failed ancestor** is *primary*; one with a failed ancestor is *secondary* (a consequence) |
| 3 | The **root cause** is the primary failure that affected the most events; ties break on depth, then timestamp, then event id |
| 4 | A failure followed by a **successful retry** is *recovered* — reported, but not an unrecovered root cause |
| 5 | A trace without failures yields `no_failure`; a trace whose failures were all recovered yields `recovered` — never a guess (D6) |
| 6 | **Affected events** are the non-failed descendants of the root cause |
| 7 | The **evidence path** is the shortest structural path from the root cause to the deepest failure |
| 8 | When the deepest failure has no failed ancestor, the closest ancestor whose recorded output reports a problem is reported as `suspected_upstream` (verdict `insufficient_evidence`) |
| 9 | **Confidence** is deterministic: `1.0` = chain of ≥ 2 with an exception type, `0.8` = chain of ≥ 2, `0.5` = suspected upstream, `0.3` = terminal failure without causal context, `0.0` = no failure |
| 10 | **Category** comes from the recorded fields only: `timeout`, `invalid_input`, `tool_error`, `llm_error`, else `unknown` |

**Grounded explanation.** The explanation is templated
(`agentlens.causal.explain.explain`) and may only interpolate recorded values
— event type, status, error message, exception type, event ids (D7). When a
failed event's payload was not retained (adaptive observation), the
explanation says *"no error detail was retained"* and the confidence is
capped at `0.5` instead of inventing a cause (D5).

**HTTP surface:**

- `GET /api/traces/{trace_id}/root-cause` → the diagnosis
- `GET /api/traces/{trace_id}` → additive `root_cause` field
  (opt out with `?include_root_cause=false`)

Both endpoints fail open: a diagnosis that cannot be produced is omitted and
logged, never raised as a 5xx (D8).

**Example:**

```
ROOT CAUSE:
  tool_call: search (event_type=tool_call, status=failure)
  error message "search backend unavailable", exception type RuntimeError

EVIDENCE PATH:
  tool_call: search
      ↓
  final_response
```

### G. Causal Visualization

*Phase 11 feature:*

The dashboard renders the graph of §E and the diagnosis of §F as one panel
(`frontend/app.js`, no build step and no new dependency).

**Deterministic layered layout.** Every node is placed in the column of its
parent — `parent_id` first, the `parent_of` edge as a fallback, a parent that
is not part of the graph ignored (never invented, D9). Inside a column the
rows are ordered by `(timestamp, event_id)`. The same graph therefore always
draws the same picture: no physics, no randomness, no layout jitter between
two renders of the same trace.

**What the panel shows:**

| Element | Meaning |
|---|---|
| Diagnosis strip | verdict, confidence, category, grounded explanation, evidence path |
| Red ring + `ROOT CAUSE` tag | the event the analyzer named |
| Red edges | the evidence path from the root cause to the deepest failure |
| Dashed red edges | `influenced` — failure propagation |
| Dashed amber / dotted blue | `retry_of` / `depends_on` |
| Red node | a failed step |

**Graceful degradation.** A trace recorded before Phase 11 has no graph: the
section is hidden rather than shown empty, and the rest of the dashboard is
unaffected. Labels are HTML-escaped.

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
│       ├── observation/        # (Phase 10+) Importance scoring & policy
│       ├── causal/             # (Phase 11+) Causal graph & root-cause analysis
│       ├── analysis/           # (Phase 4+) Failure diagnosis
│       └── dashboard/          # (Phase 5+) Web API & frontend
├── examples/
│   └── simple_agent.py         # Example agent
├── frontend/                   # Vanilla-JS dashboard (no build step)
│   ├── index.html              # Dashboard
│   ├── app.js                  # Rendering + causal panel
│   ├── style.css               # Styles
│   └── test.html               # Browser test harness
├── tests/
│   ├── test_*.py               # Unit & integration tests
│   ├── causal_scenarios.py     # (Phase 11+) Labelled evaluation scenarios
│   └── benchmark_*.py          # (Phase 10+) Measurement harnesses
├── docs/
│   ├── architecture.md         # This file
│   └── phase*.md               # Phase details
├── pyproject.toml              # Project metadata & dependencies
├── README.md                   # Quick start
└── .gitignore                  # Git ignore rules
```

---

**Project Status:** Phase 11 — Causal Analysis & Root-Cause Diagnosis
**Last Updated:** 2026-10-22
