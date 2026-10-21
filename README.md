# AgentLens

**Adaptive Causal Observability for Cost-Efficient Debugging of Autonomous AI Agents**

A research project investigating how to minimize observability overhead while
preserving accurate diagnosis of AI-agent failures. Records every agent step
(LLM calls, tool calls, retrieval, memory, decisions, errors, retries) with
parent links, builds a causal graph, and names the root cause of failures — all
in a zero-build-step browser dashboard.

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/Beastdzns/agentlens.git
cd agentlens
python3.12 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

# 2. Seed the demo database (separate ./demo.db — your runs are untouched)
./venv/bin/python scripts/seed_demo_data.py

# 3. Start the API + dashboard
AGENTLENS_STORAGE=sqlite AGENTLENS_DB_PATH=./demo.db ./venv/bin/python -m uvicorn agentlens.web_api:app --reload

# 4. Open http://localhost:8000/app/
```

Click a trace in the "Trace History" card to see its events, causal graph, and
root-cause diagnosis. Run your own query via `POST /api/run` — a failing run
returns the failed trace (`status="failure"`, `trace_id`, events, error message,
causal analysis) instead of a bare 502.

## What it does

- **Trace capture** — every pipeline step is recorded with input/output, latency,
  parent links, and failure metadata (`capture_llm_call`, `capture_tool_call`,
  `capture_step`).
- **Causal analysis** — turns events into a causal graph (`parent_of`,
  `retry_of`, `depends_on`, `influenced`) and a rule-based root-cause diagnosis
  (verdicts: `root_cause`, `no_failure`, `recovered`, `insufficient_evidence`).
- **Dashboard** — trace history with status/date filters and keyset pagination,
  event timeline, failure summary banner, causal graph visualization; responsive
  layout (mobile-friendly).
- **Demo data** — `scripts/seed_demo_data.py` seeds 4 traces (success,
  retry-recovered partial, DB failure, Gemini-quota failure) into `./demo.db`
  (`AGENTLENS_DB_PATH` selects the file).

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health |
| GET | `/` | Service information |
| POST | `/api/run` | Run an agent query; returns the trace (failed runs return `status="failure"`, not 502) |
| GET | `/api/traces` | List traces: status + date filters, keyset pagination |
| GET | `/api/traces/{trace_id}` | Retrieve a trace (+ `causal_graph` and `root_cause`) |
| GET | `/api/traces/{trace_id}/causal-graph` | Causal graph of a trace |
| GET | `/api/traces/{trace_id}/root-cause` | Root-cause diagnosis |

## Tests

```bash
# Python suite
./venv/bin/python -m pytest -q            # 616 tests

# Browser suite — open in any browser
xdg-open frontend/test.html               # 227 tests

# Lint
ruff check src/ tests/ examples/          # no new findings vs. baseline
```

## Phase status

| Phase | Focus | Status |
|-------|-------|--------|
| 1–5 | Events, SDK collector, storage, Gemini demos, tests | Complete |
| 6 | FastAPI API and simple web dashboard | Complete |
| 7 | Live trace visualization (timeline, tree, filters) | Complete |
| 8 | Failure tracking (capture, presentation, testing) | Complete |
| 9 | Persistent storage (SQLite) | Complete |
| 10 | Adaptive observability | Complete |
| 11 | Causal analysis and root-cause diagnosis | Complete |
| 12 | Final web application (local demo only): failed-run trace, trace history, responsive UI, seeded demo data | Complete |

Phase 12 is **local-demo-only**: no authentication, no Docker/hosting, no CI, no
multi-user. LangChain / LlamaIndex normalizer adapters remain stubs (planned
work, out of Phase 12 scope). Full design notes live in `docs/architecture.md`
and per-phase write-ups under `docs/`.

**Status:** Phase 12 — Final Demo Complete  
**Last Updated:** 2026-10-22
