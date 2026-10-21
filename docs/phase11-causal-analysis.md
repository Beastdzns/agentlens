# Phase 11 — Causal Analysis & Root-Cause Diagnosis

**Status:** Complete (merged into `main`)

Phase 10 made traces *cheap*. Phase 11 makes them *readable*: AgentLens now
builds an explicit causal graph of a trace, names the root cause of a failure
with a grounded explanation, draws the graph in the dashboard, and measures how
often that diagnosis is right.

```
events  ->  build_graph()  ->  CausalGraph  ->  RootCauseAnalyzer.analyze()  ->  RootCause
           (pure, D2)         nodes/edges      (rules only, D1)              verdict + evidence
                                                 |
                                                 +-> explain()  ->  grounded text  ->  API  ->  dashboard SVG
```

---

## 1. What was built

| Member | Branch | Scope |
|---|---|---|
| V — integrator | `feature/causal-graph` | `agentlens.causal.graph`, `GET /api/traces/{id}/causal-graph`, `TraceResponse.causal_graph` |
| S — rules | `feature/root-cause-analyzer` | `agentlens.causal.analyzer`, `agentlens.causal.explain`, `GET /api/traces/{id}/root-cause`, `TraceResponse.root_cause` |
| N — research | `research/causal-evaluation` | labelled scenario set, evaluation suite, benchmark harness, dashboard panel, this write-up |

### 1.1 The causal graph — `agentlens.causal.graph`

`build_graph(events)` is a **pure** function: no I/O, no mutation of its input,
and the same events always produce the same graph (D2).

**Node** — one event: identity, status, timings, observation fields, a short
label and, when recorded, the error message and exception type.

**Edges** — a closed taxonomy of four relations:

| Relation | Direction | Derived from | Traversed? |
|---|---|---|---|
| `parent_of` | parent → child | `event.parent_id` | yes |
| `retry_of` | failed attempt → retry | `metadata["retry_of"]` | yes |
| `depends_on` | node → dependency | `metadata["depends_on"]` | yes |
| `influenced` | failed node → child | a failing parent (annotation) | no |

`caused` is deliberately **not** a construction edge — it is the analyzer's
*verdict*, never one of its inputs (D3). `influenced` duplicates the
`parent_of` direction and exists only so the UI can draw failure propagation
differently; it never creates ancestry.

**Guarantees:** deterministic ordering (nodes by `(timestamp, event_id)`, edges
by `(relation, source, target)`), grounded edges (a parent or `retry_of` id
outside the trace is ignored, never invented as a phantom node — D9), graceful
with missing evidence (a dropped event keeps its node and its parent link, so
the structure survives adaptive observation), and cycle-safe iterative
traversal.

### 1.2 The analyzer — `agentlens.causal.analyzer`

`RootCauseAnalyzer.analyze(graph)` walks the graph with **rules only** — no
model is ever called, so the same trace always yields the same diagnosis
(D1, NFR6).

| # | Rule |
|---|---|
| 1 | A **failure** is a node with status `failure` (an `error`-typed node counts too) |
| 2 | A failure with **no failed ancestor** is *primary*; one with a failed ancestor is *secondary* |
| 3 | The **root cause** is the primary failure that affected the most events; ties break on depth, then timestamp, then event id |
| 4 | A failure followed by a **successful retry** is *recovered* — reported, but not an unrecovered root cause |
| 5 | A trace without failures yields `no_failure`; a trace whose failures were all recovered yields `recovered` — never a guess (D6) |
| 6 | **Affected events** are the non-failed descendants of the root cause |
| 7 | The **evidence path** is the shortest structural path from the root cause to the deepest failure |
| 8 | When the deepest failure has no failed ancestor, the closest ancestor whose own output reports a problem is reported as `suspected_upstream` (verdict `insufficient_evidence`) |
| 9 | **Confidence** is deterministic: `1.0` = chain of ≥ 2 with an exception type, `0.8` = chain of ≥ 2, `0.5` = suspected upstream, `0.3` = terminal failure without causal context, `0.0` = no failure |
| 10 | **Category** comes from the recorded fields only: `timeout`, `invalid_input`, `tool_error`, `llm_error`, else `unknown` |

Verdicts: `root_cause` / `no_failure` / `recovered` / `insufficient_evidence`.

**Grounded explanation.** `explain()` is templated and may only interpolate
recorded values — event type, status, error message, exception type, event ids
(D7). When a failed event's payload was not retained (adaptive observation),
the explanation says *"no error detail was retained"* and the confidence is
capped at `0.5` instead of inventing a cause (D5).

### 1.3 The dashboard panel

The dashboard shows a **Causal Analysis** section for every trace that has a
graph:

- a diagnosis strip (verdict, confidence, category, explanation, evidence path);
- a deterministic layered SVG of the graph — every node sits in the column of
  its parent, the rows inside a column are ordered by `(timestamp, event_id)`,
  so the same graph always draws the same picture;
- the root cause highlighted with a red ring and a `ROOT CAUSE` tag, the
  evidence path drawn in red, failed steps marked;
- a legend for the relation styles.

No new dependency: the layout and the SVG are ~200 lines of plain JavaScript
in `frontend/app.js`. A trace without a graph hides the section entirely
instead of showing an empty box (graceful degradation).

### 1.4 HTTP surface

| Endpoint | Returns |
|---|---|
| `GET /api/traces/{trace_id}/causal-graph` | `{trace_id, nodes[], edges[]}` |
| `GET /api/traces/{trace_id}/root-cause` | the diagnosis |
| `GET /api/traces/{trace_id}` | the trace plus the additive `causal_graph` and `root_cause` fields |

Both fields are optional and included by default; a client that does not need
them opts out with `?include_causal_graph=false` / `?include_root_cause=false`.
Every path **fails open** (D8): a graph or diagnosis that cannot be produced is
omitted and logged, never raised as a 5xx.

---

## 2. Evaluation

### 2.1 The labelled scenario set — `tests/causal_scenarios.py`

Seven hand-built traces with a known ground truth:

| Scenario | What it tests |
|---|---|
| `tool_error` | a tool call fails with an exception type |
| `timeout` | a timeout is classified as `timeout` |
| `invalid_input` | a bad request is classified as `invalid_input` |
| `retry_recovered` | the failure is recovered — the verdict is `recovered`, not a root cause |
| `bad_upstream` | the failing step's own output is fine, its parent produced nonsense → `suspected_upstream` |
| `cascade` | one failure propagates over several steps |
| `skewed_clock` | the root cause is timestamped **after** its child — chronology is misleading, structure is not |

### 2.2 H3 — causal graph vs. chronological baseline

The baseline is the naive rule "the first failure, by wall-clock". H3 asks
whether the causal graph localizes the failure better than that flat rule.

```
scenario                 graph    baseline
--------------------------------------------
tool_error                 hit         hit
timeout                    hit         hit
invalid_input              hit         hit
retry_recovered            hit        MISS
bad_upstream               hit        MISS
cascade                    hit         hit
skewed_clock               hit        MISS
--------------------------------------------
accuracy                   7/7         4/7
accuracy %              100.0%       57.1%
```

**H3 is supported on this set**: the graph localizes the root cause in 7/7
scenarios, the chronological baseline in 4/7. The three cases the baseline
misses are exactly the ones a flat log cannot express — a recovered failure, a
suspected upstream cause, and a trace whose clock is skewed relative to its
causal order.

### 2.3 Robustness — what survives adaptive observation

Two degradations are applied to every scenario:

| Degradation | What it removes |
|---|---|
| **adaptive** | the realistic policy mirror of Phase 10 — payloads compressed, failures kept in full |
| **hostile** | payloads *and* metadata dropped — the `retry_of` link is lost with them |

```
scenario            full    adaptive     hostile   adaptive conf
------------------------------------------------------------
tool_error           hit         hit         hit            1.00
timeout              hit         hit         hit            1.00
invalid_input        hit         hit         hit            0.30
retry_recovered      hit         hit        MISS            0.00
bad_upstream         hit        MISS        MISS            0.30
cascade              hit         hit         hit            0.80
skewed_clock         hit         hit         hit            0.80
------------------------------------------------------------
adaptive mode keeps the root cause in 6/7 scenarios
```

The structural skeleton survives adaptive observation, so 6 of 7 diagnoses are
unchanged. The two degradations are visible and honest:

- `bad_upstream` needs the parent's *output payload* to spot that the parent
  produced nonsense; under adaptive observation that payload is compressed
  away, and the analyzer reports `insufficient_evidence` with confidence `0.30`
  instead of guessing (D5).
- `retry_recovered` under the hostile degradation loses the `retry_of` link, so
  the recovery can no longer be proven — the analyzer falls back to reporting
  the failure rather than claiming it was recovered.

This is the intended behaviour: **the analyzer degrades into a lower-confidence
answer, never into a wrong confident one.**

### 2.4 Explanation quality

| Metric | Result |
|---|---|
| Explanations that quote the recorded error message | 7/7 |
| Degraded explanations that say *"no error detail was retained"* | 7/7 |
| Mean explanation length | 265 characters |

Every explanation is interpolating recorded values only. Nothing is invented,
and the absence of evidence is stated as such.

### 2.5 Latency (NFR1)

Graph construction + diagnosis, on one machine, mean of 5 runs:

| Events | Build (ms) | Analyze mean (ms) | Analyze p95 (ms) |
|---:|---:|---:|---:|
| 100 | 1.21 | 0.10 | 0.10 |
| 1,000 | 10.18 | 0.84 | 0.90 |
| 10,000 | 119.12 | 8.01 | 8.09 |

The diagnosis itself is linear-ish and stays in single-digit milliseconds even
for a 10,000-event trace. Graph *construction* is the dominant cost at that
size (119 ms) — it is a one-off per trace request and is cached by the
client, but it is the number to watch if very large traces become common.

> An earlier implementation of the analyzer was quadratic (565 ms for 1,000
> events). It was rewritten to a single memoised pass over the structural
> edges; `CausalGraph.depth()` was memoised at the same time and now matches
> its documented "longest chain" semantics.

### 2.6 Reproducing

```bash
./venv/bin/python tests/benchmark_causal.py     # the numbers above
./venv/bin/python -m pytest -q                  # the whole suite
```

The evaluation itself is a normal test module, so a regression in the
diagnosis fails the build:

```bash
./venv/bin/python -m pytest tests/test_causal_evaluation.py -q
```

---

## 3. Tests

| Suite | Tests |
|---|---|
| `tests/test_causal_graph.py` | 39 |
| `tests/test_root_cause.py` | 39 |
| `tests/test_causal_evaluation.py` | 76 |
| `tests/integration/test_causal_integration.py` | 26 |
| **Phase 11 total** | **180** |
| Whole Python suite | 591 |
| Browser suite (`frontend/test.html`) | 162 |

The browser suite covers the layout (columns, deterministic ordering, cycle
safety, orphans), the SVG (nodes, edges, root-cause highlight, escaping), the
diagnosis strip and the show/hide behaviour of the panel.

---

## 4. Known limitations

- The analyzer is rule-based by design (D1). It does not interpret semantics:
  a step that returned a wrong answer without raising an error is only caught
  through the `suspected_upstream` rule, and only if its payload was retained.
- A graph is rebuilt per request. For very large traces this is visible
  (§2.5); caching it per trace would remove the cost from the request path.
- The layout is a layered left-to-right tree. Wide traces become very wide and
  scroll horizontally; a hierarchical (Sugiyama-style) layout would be
  denser but is a much larger piece of work.
- The evaluation set is seven hand-built scenarios. It is deliberately
  labelled and reproducible, but it is small, and a larger corpus of real
  traces would be needed to claim general accuracy.
