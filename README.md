# AgentLens

**Adaptive Causal Observability for Cost-Efficient Debugging of Autonomous AI Agents**

A research project investigating how to minimize observability overhead while preserving accurate diagnosis of AI-agent failures.

## Quick Start

### Prerequisites

- Python 3.10 or later
- pip or uv
- A Gemini API key for live LLM calls (optional; offline fallbacks are available)

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

    This installs the AgentLens SDK, FastAPI server, Gemini client, in-memory
    storage, and development tools.

### Running Tests

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src/agentlens --cov-report=html
```

The current suite contains **411 Python tests** (events, collector, storage,
web API, failure capture, failure tracking, adaptive observability,
integration). The frontend also ships a zero-build JavaScript test harness:

```bash
# Open in a browser and read the summary at the top of the page
xdg-open frontend/test.html        # 85 browser tests
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
**Option 3 — Direct Uvicorn command**
├── src/agentlens/
│   ├── events.py           # Event models and schemas
│   ├── collector.py        # Synchronous collector compatibility API
│   ├── sdk_collector/      # Async collector, context, normalizer
│   ├── observation/        # Adaptive observability: scoring + policy
│   ├── causal/             # Causal graph + root-cause analysis
│   ├── storage/            # In-memory and SQLite backends
│   └── web_api.py          # FastAPI app
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

### CORS — development setup

| Phase | Focus | Status |
|-------|-------|--------|
| 1-5 | Events, SDK collector, storage, Gemini demos, tests | Complete |
| 6 | FastAPI API and simple web dashboard | Complete |
| 7 | Live trace visualization (timeline, tree, filters) | Complete |
| 8 | Failure tracking (capture, presentation, testing) | Complete |
| 9 | Persistent storage | Complete |
| 10 | Adaptive observability | Complete |
| 11 | Causal analysis and root-cause diagnosis | Complete |
| 12 | Final web application (local demo): failed-run trace, trace history, responsive UI, seeded demo data | Complete |

## Research Questions

**Primary Question:**
Can adaptive observability reduce telemetry and system overhead while preserving accurate and fast diagnosis of AI-agent failures?

**Key Hypotheses:**
- H1: Adaptive tracing reduces telemetry volume vs. full tracing
- H2: Adaptive tracing preserves failure-diagnosis accuracy
- H3: Causal graphs improve root-cause localization
- H4: High-fanout workloads expose greater overhead

## Technology Stack

**Backend:**
- Python 3.10+
- FastAPI and Uvicorn
- Pydantic (data validation)
- asyncio (async execution)
- `google-genai` (Gemini API)

**Frontend:**
- Vanilla HTML, CSS, and JavaScript
- No frontend build step required

**Storage:**
- Async in-memory event and trace stores
- PostgreSQL or another persistent backend planned

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

**Status:** Phase 8 - Failure Tracking Complete
**Last Updated:** 2026-10-17

---

## SDK Collector Module (`sdk_collector/`)

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

### Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the project in editable mode (includes all dev dependencies)
pip install -e ".[dev]"
```

---

## Current Limitations and Next Steps

- The current stores are in memory and reset when the server restarts.
- LangChain and LlamaIndex normalizer adapters remain stubs.
- Causal graph generation and automatic root-cause analysis are planned.
- The dashboard currently uses vanilla JavaScript to keep the demo build-free.

### Planned Work

- **LangChain adapter** — `EventNormalizer.from_langchain_event()` is a stub.
- **LlamaIndex adapter** — `EventNormalizer.from_llamaindex_event()` is a stub.
- **Causal graph builder** — Phase 3: build Neo4j graph from parent/child event links.



---

## Frontend — AgentLens Observability Dashboard

A zero-build-step single-page application that lets you run agent queries
and inspect the resulting telemetry (stat cards, event timeline, JSON
input/output blocks) directly in a browser.

### Files

```
frontend/
+-- index.html   # App shell & layout
+-- style.css    # Full design system (dark mode, tokens, components)
+-- app.js       # All logic — API client, mock mode, render engine
```

### Launching locally (quickest way)

**Option 1 — Single unified server (API + frontend)**

For offline fallback mode:

```bash
source venv/bin/activate
python run_demo.py
```

For live Gemini execution:

```bash
source venv/bin/activate
cp .env.example .env
# Edit .env and set GEMINI_API_KEY and GEMINI_MODEL
python run_demo.py
```

The launcher prints the actual dashboard URL and automatically chooses the
next available port if the default port is occupied. Open the printed `/app/`
URL in your browser.

**Option 2 — Standalone frontend server**

```bash
# Terminal 1: Backend
uvicorn agentlens.web_api:app --reload --port 8000

# Terminal 2: Frontend
python -m http.server 5500 --directory frontend
# then open http://localhost:5500
```

### Connecting to the FastAPI backend

The `API_BASE_URL` constant at the top of `app.js` automatically adapts:
- If hosted via the unified backend on port 8000, it uses `http://localhost:8000`.
- If served from a separate server (e.g. port 5500), it targets `http://localhost:8000` via CORS.

The frontend is served by `agentlens.web_api` and expects these endpoints:

| Method | Path      | Description                         |
|--------|-----------|-------------------------------------|
| GET    | /health   | Check server health                |
| GET    | /         | Service information                |
| POST   | /api/run  | Run an agent query, return a trace |
| GET    | /api/traces | List stored traces (filters, date range, keyset pagination) |
| GET    | /api/traces/{trace_id} | Retrieve a stored trace |

**Request body:**
```json
{ "query": "How can AI agents be debugged?" }
```

**Response body (fields the UI reads):**
```json
{
  "trace_id":        "uuid-string",
  "response":        "Agent's final answer",
  "status":          "success | failure | partial",
  "total_latency_ms": 342.7,
  "events": [
    {
      "event_id":    "uuid",
      "parent_id":   "uuid or null",
      "agent_id":    "string",
      "event_type":  "llm_call | tool_call | retrieval | ...",
      "status":      "success | failure | pending | partial",
      "latency_ms":  45.2,
      "input_data":  {},
      "output_data": {},
      "metadata":    {},
      "timestamp":   "ISO-8601"
    }
  ]
}
```

> **Note:** the UI gracefully handles both `latency_ms` and `latency` field
> names on events, so either serialization from `AgentEvent.to_dict()` works.

### CORS — development setup

The API already enables CORS. Set `CORS_ORIGINS` to a comma-separated list of
allowed origins when running the frontend separately. The default is `*` for
local development.

### Mock Mode — test all visual states without a backend

Click the **"Live Mode"** toggle in the top-right header to switch to
**Mock Mode**. Four pre-built scenarios appear in the left sidebar:

| Scenario          | What it demonstrates                         |
|-------------------|----------------------------------------------|
| Clean Success Run | All-green event timeline, full response       |
| Partial Failure   | Mixed green/amber badges, retry event         |
| Full Failure      | Red trace status, error events, empty response|
| Backend Error 500 | HTTP error banner, no results rendered        |

---

## Phase 8 — Failure Tracking

Phase 8 makes agent failures first-class: every LLM/tool step is captured with
its exception details, the failure is presented in the dashboard with its
context (parent step, retry chain), and the behaviour is covered by scenario
tests.

### Automatic failure capture (backend)

Three async context managers on `EventCollector`
(`src/agentlens/sdk_collector/collector.py`) wrap each pipeline step. They time
the step, record `SUCCESS` on clean exit, and on **any** exception record a
`FAILURE` event **before re-raising**, so a failed run always leaves a
persisted, identifiable record behind.

| Wrapper | Records | Used for |
|---|---|---|
| `capture_step(event_type, …)` | Generic step | retrieval, memory write, final response |
| `capture_llm_call(prompt, model, …)` | `LLM_CALL` with prompt + response | planning, synthesis |
| `capture_tool_call(tool_name, …)` | `TOOL_CALL` with input + output | web search and other tools |

```python
async with collector.capture_llm_call(
    prompt="Summarise the evidence",
    model="gemini-2.5-flash",
    metadata={"step": "synthesis"},
) as result:
    result[0] = await call_llm(prompt)   # response captured automatically
```

Every captured failure event carries this metadata:

```json
{
  "step": "synthesis",
  "exception_type": "TimeoutError",
  "exception_message": "Gemini request timed out after 30s"
}
```

**Retry linkage** — pass `retry_of=<failed_event_id>` and an incremented
`attempt` when retrying a step; both are stored in the new event's metadata so
the UI can render the retry chain:

```python
async with collector.capture_llm_call(
    prompt=prompt, retry_of=failed_event_id, attempt=2,
) as result:
    result[0] = await call_llm(prompt)
```

### Failure presentation (frontend)

When a trace contains failed events the dashboard:

- Shows a **failure summary banner** above the timeline with the failed step,
  the error message (`output_data.error` / `metadata.exception_message`) and
  the exception type.
- Defaults the **status filter to “Failure”** so the culprit is immediately
  visible, with a one-click “Show full trace” return.
- Renders a **parent breadcrumb** and **retry badges** (`↩ retry of`,
  `↻ retried by`) inside each failed event's detail block.
- Falls back to scanning event statuses when the API's `failure_event_ids`
  field is absent.

### Failure testing

| Test file | Tests | Covers |
|---|---|---|
| `tests/test_failure_capture.py` | 18 | Wrapper success/failure paths, re-raise, retry metadata |
| `tests/test_failure_tracking.py` | 26 | Timeout, invalid-input, tool-failure, retry and exact-identification scenarios |
| `frontend/test.js` | +23 assertions | Failure banner, parent breadcrumb, retry badges, smart-filter default |

The trace endpoint exposes the failure summary the UI consumes:

```json
{
  "trace_id": "uuid",
  "status": "failure",
  "failure_count": 1,
  "failure_event_ids": ["uuid-of-the-failed-event"]
}
```

> **Note:** A live run that fails is no longer a bare HTTP 502 (fixed in Phase
> 12): the API returns the failed trace — `status="failure"`, its `trace_id`,
> its events, the error message, and the causal analysis — so the dashboard
> can show *why* the run failed instead of a generic error banner.  Viewing a
> failed trace directly via `GET /api/traces/{trace_id}` still works, as does
> the **Full Failure** mock scenario.


---

## Phase 10 — Adaptive Observability

Phase 10 makes tracing **adaptive**: every event gets a deterministic
importance score, and the payload is kept, compressed, reduced to metadata or
dropped according to that score — without ever degrading the evidence needed
to diagnose a failure.

### How it works

```
Event → ImportanceScorer.score() → ObservationPolicy.decide() → ObservationPolicy.transform() → store
```

| Score | Observation level | Payload kept |
|---|---|---|
| `>= 0.60` | `full` | everything |
| `>= 0.35` | `compressed` | truncated to `AGENTLENS_PAYLOAD_MAX_BYTES` (default 4 KB) |
| `>= 0.15` | `metadata_only` | `input_data` / `output_data` nulled |
| `< 0.15` | `drop` | payload nulled, `metadata = {"_dropped": true}` |

`status == FAILURE` and the event types `ERROR`, `RETRY`, `FINAL_RESPONSE` are
**always** observed in full, whatever their score. Dropped events keep their
`event_id`, `trace_id`, `parent_id`, type, status, timestamp and latency, so
the causal graph stays intact.

### Enabling it

Adaptive observability is **off by default** (`full` mode is a no-op). Turn it
on with environment variables:

```bash
export AGENTLENS_OBSERVATION_POLICY=adaptive
# optional tuning
export AGENTLENS_COMPRESS_THRESHOLD=0.60
export AGENTLENS_METADATA_THRESHOLD=0.35
export AGENTLENS_DROP_THRESHOLD=0.15
export AGENTLENS_PAYLOAD_MAX_BYTES=4096
```

Every event returned by the API then carries `importance_score` and
`observation_level`, and a run response carries an `observation_summary`
(count per level). The dashboard renders an observation badge and an
importance bar on each event card.

### Measured effect

`tests/benchmark_observation.py` processes a 10,000-event synthetic trace
through both modes (payload bytes **and** on-disk SQLite size are measured):

| Metric | Full | Adaptive | Change |
|---|---|---|---|
| Events retained | 10,000 | 10,000 | unchanged |
| Payload size | 10,887.6 KB | 617.3 KB | **−94.3 %** |
| SQLite file size | 15,612 KB | 4,480 KB | **−71.3 %** |
| Processing time | 0.106 s | 0.112 s | **+5.4 %** (≈ 0.6 µs/event) |
| Diagnosis preserved | yes | yes | — |

### Tests

| Test file | Covers |
|---|---|
| `tests/test_importance_scoring.py` | scoring rules, modifiers, repetition penalty, clamping |
| `tests/test_observation_policy.py` | threshold routing, D4 override, transformation, fail-open |
| `tests/test_adaptive_evaluation.py` | evaluation invariants (evidence preserved, skeleton retained) |
| `tests/integration/test_observation_integration.py` | end-to-end: API surface, storage, SQLite round-trip, determinism |

Full details and the evaluation write-up: [docs/phase10-adaptive-observability.md](docs/phase10-adaptive-observability.md).


---

## Phase 11 — Causal Analysis & Root-Cause Diagnosis

Phase 11 turns a flat list of events into a **causal graph**, names the **root
cause** of a failure with a grounded explanation, and draws the result in the
dashboard.

### How it works

```
events → build_graph() → CausalGraph → RootCauseAnalyzer.analyze() → RootCause
                        (pure)         (rules only)                 verdict + evidence
```

The graph uses a closed set of relations — `parent_of`, `retry_of`,
`depends_on` and `influenced` (an annotation for failure propagation). It is
built by a pure function: the same events always produce the same graph, and a
parent id that points outside the trace is ignored rather than invented.

The analyzer is **rule-based** — no model is called — so the same trace always
yields the same diagnosis. It reports one of four verdicts:

| Verdict | Meaning |
|---|---|
| `root_cause` | a failure with a named cause, an evidence path and a confidence |
| `no_failure` | the trace has no failed step |
| `recovered` | every failure was followed by a successful retry |
| `insufficient_evidence` | the cause cannot be pinned down — the analyzer says so instead of guessing |

Explanations are templated and quote only recorded values. When the payload was
not retained (adaptive observation), the explanation says *"no error detail was
retained"* and the confidence is capped instead of a cause being invented.

### API

| Endpoint | Returns |
|---|---|
| `GET /api/traces/{trace_id}/causal-graph` | the graph (`nodes`, `edges`) |
| `GET /api/traces/{trace_id}/root-cause` | the diagnosis |
| `GET /api/traces/{trace_id}` | the trace plus the `causal_graph` and `root_cause` fields |

The two extra fields are optional and included by default; opt out with
`?include_causal_graph=false` / `?include_root_cause=false`. Every path fails
open: a graph or diagnosis that cannot be produced is omitted and logged, never
returned as a server error.

### Dashboard

Traces with a graph show a **Causal Analysis** section: the diagnosis on top,
then a deterministic layered SVG of the graph with the root cause highlighted
and the evidence path drawn in red. No new dependency, no build step. Traces
without a graph hide the section entirely.

### Measured effect

`tests/benchmark_causal.py` compares the analyzer with the naive "first failure
by wall-clock" baseline on seven labelled scenarios:

| Metric | Causal graph | Chronological baseline |
|---|---|---|
| Root cause localized | **7/7** | 4/7 |
| Root cause survives adaptive observation | **6/7** | — |
| Explanations quoting the recorded error | **7/7** | — |
| Analysis of a 10,000-event trace | 8.0 ms mean / 8.1 ms p95 | — |

### Tests

| Test file | Covers |
|---|---|
| `tests/test_causal_graph.py` | construction, edge taxonomy, determinism, traversal |
| `tests/test_root_cause.py` | analyzer rules, verdicts, confidence, explanations |
| `tests/test_causal_evaluation.py` | the labelled scenario set, H3, degradation, perf |
| `tests/integration/test_causal_integration.py` | end-to-end: both endpoints, opt-out, fail-open |

Full details and the evaluation write-up: [docs/phase11-causal-analysis.md](docs/phase11-causal-analysis.md).


---

## Phase 12 — Final Web Application (Local Demo Only)

Phase 12 is a **local-demo-only** release: no authentication, no Docker, no
hosting, no CI, no multi-user — just one command to run, seeded data to
browse, and a solid dashboard.

### One-command local demo

```bash
# 1. Seed the demo database (a separate ./demo.db — your real runs are untouched)
./venv/bin/python scripts/seed_demo_data.py

# 2. Start the API + dashboard against the seeded database
AGENTLENS_STORAGE=sqlite AGENTLENS_DB_PATH=./demo.db ./venv/bin/python -m uvicorn agentlens.web_api:app --reload

# 3. Open http://localhost:8000/app/
```

The dashboard loads the "Trace History" card listing the seeded runs
(success, retry-recovered, and two failures).  Click any trace to see its
events, causal graph, and root-cause diagnosis.  Run your own query with
`POST /api/run` — a failing run returns the failed trace (not a 502) with the
error message and the causal analysis attached.

### What changed in Phase 12

- **Backend (`feature/final-backend`)**
  - `POST /api/run` returns the failed trace (`status="failure"`, `trace_id`,
    events, error message) instead of a bare HTTP 502; `RunResponse` gains the
    optional `error`, `causal_graph` and `root_cause` fields.
  - The four duplicate step events are gone — a live run records 7 events,
    not 11 — and the pipeline's parent chain is connected
    (`decision → search → retrieval → memory-write → synthesis → final
    response`), so every live trace draws a real causal graph.
  - `capture_step` records its output like `capture_llm_call` /
    `capture_tool_call` already did; `EventCollector.last_event` links
    parent steps.
- **Frontend (`feature/final-frontend`)**
  - Trace history panel against `GET /api/traces`: status + date-range
    filters, keyset "Load more" pagination.
  - Responsive layout — the first `@media` breakpoints (`style.css` had
    none): stacked sidebar/results on narrow windows, compact stats and
    headers on phones.
  - Live-run failure state: error message, root-cause verdict/explanation,
    and buttons to jump to the causal analysis or the failed steps.
- **Demo (`research/final-demo`)**
  - `scripts/seed_demo_data.py` writes four demo traces into `./demo.db`
    (selectable via `AGENTLENS_DB_PATH`).
  - Demo guide above; both suites stay green (616 Python tests,
    227 browser tests).

### Still out of scope (local-demo-only)
- Authentication, multi-user, Docker, CI/CD, hosting (explicitly dropped for
  Phase 12).
- LangChain / LlamaIndex normalizer adapters (`EventNormalizer.from_*`
  remain stubs).
- Persistent storage beyond SQLite for the local demo.

---

**Status:** Phase 12 — Final Demo Complete  
**Last Updated:** 2026-10-22

