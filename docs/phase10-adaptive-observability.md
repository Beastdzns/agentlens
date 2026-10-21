# Phase 10 — Adaptive Observability

**Status:** Complete (merged into `main`)

Adaptive observability keeps the evidence needed to diagnose an agent while
cutting how much payload is written, stored and shipped to the dashboard.

```
Event  ->  ImportanceScorer.score()  ->  ObservationPolicy.decide()  ->  ObservationPolicy.transform()  ->  store
             deterministic [0..1]           FULL / COMPRESSED /             payload kept / truncated /
                                           METADATA_ONLY / DROP           nulled / skeleton only
```

---

## 1. Design

### 1.1 Importance scoring — `agentlens.observation.importance`

Rule-based, deterministic, pure (no storage or network access):

```
raw = (base_by_status + weight_by_type) / 2 + modifiers      ->  clamp to [0, 1]
```

| Contribution | Value |
|---|---|
| `base_by_status` | FAILURE `0.90`, PARTIAL `0.70`, PENDING `0.60`, SUCCESS `0.30` |
| `weight_by_type` | ERROR `1.00`, FINAL_RESPONSE `0.95`, RETRY `0.85`, DECISION `0.80`, SUBAGENT_* `0.70`, TOOL_CALL `0.55`, LLM_CALL / RETRIEVAL `0.50`, MEMORY_* `0.35` |
| `parent_bonus` | `+0.10` when `parent_id is None` (structural root) |
| `latency_bonus` | `+0.10` when `latency_ms >= 1000` |
| `exception_bonus` | `+0.10` when `metadata["exception_type"]` is set |
| `repetition_penalty` | `−0.15 × min(repeat_index, 3)` for repeated **successful** steps; `FINAL_RESPONSE` and `DECISION` are exempt, failures are never penalised |

The two base contributions are **averaged, not summed**, so each behaves as an
independent `0..1` axis. This keeps the score on the same scale as the frozen
policy thresholds below — with a summed score almost every event would clear
`0.60` and adaptive mode would degenerate into a no-op.

### 1.2 Observation policy — `agentlens.observation.policy`

| Score | Level | Payload kept |
|---|---|---|
| `>= 0.60` | `FULL` | everything |
| `>= 0.35` | `COMPRESSED` | payload truncated to `payload_max_bytes` (default 4 KB) with a `_truncated` summary |
| `>= 0.15` | `METADATA_ONLY` | `input_data` / `output_data` nulled |
| `< 0.15` | `DROP` | payload nulled, `metadata = {"_dropped": true}` |

**Safety override (D4).** `status == FAILURE` and the event types `ERROR`,
`RETRY`, `FINAL_RESPONSE` are **always** `FULL`, whatever their score. This is
what makes adaptive mode safe to enable: the evidence a diagnosis depends on
is never compressed away.

**Skeleton retention (D5).** A dropped event keeps its `event_id`, `trace_id`,
`parent_id`, `event_type`, `status`, `timestamp` and `latency_ms`. The causal
graph of Phase 11 can therefore be built from an adaptive trace.

**Full mode is a no-op (D6).** `AGENTLENS_OBSERVATION_POLICY=full` (the
default) neither scores nor transforms anything.

### 1.3 Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AGENTLENS_OBSERVATION_POLICY` | `full` | `full` (no-op) or `adaptive` |
| `AGENTLENS_COMPRESS_THRESHOLD` | `0.60` | score ≥ → `FULL` |
| `AGENTLENS_METADATA_THRESHOLD` | `0.35` | score ≥ → `COMPRESSED` |
| `AGENTLENS_DROP_THRESHOLD` | `0.15` | score ≥ → `METADATA_ONLY`, else `DROP` |
| `AGENTLENS_PAYLOAD_MAX_BYTES` | `4096` | COMPRESSED truncation threshold |

---

## 2. Evaluation

### 2.1 Method

`tests/benchmark_observation.py` builds one synthetic trace of 10,000 events —
a deep, repetitive agent run that ends in a terminal failure — and processes it
through both modes:

```bash
./venv/bin/python tests/benchmark_observation.py
```

Every event is scored, routed, transformed and then written to a real SQLite
database, so storage numbers are measured on disk rather than estimated.

### 2.2 Results (10,000 events)

| Metric | Full (baseline) | Adaptive | Change |
|---|---|---|---|
| Events retained | 10,000 | 10,000 | unchanged |
| Payload size | 10,887.6 KB | 617.3 KB | **−94.3 %** |
| SQLite file size | 15,612 KB | 4,480 KB | **−71.3 %** |
| Processing time | 0.106 s | 0.112 s | **+5.4 %** (≈ 0.6 µs/event) |
| Diagnosis preserved | yes | yes | — |

Level distribution in adaptive mode:

| Level | Events |
|---|---|
| `full` | 1 (the terminal failure) |
| `compressed` | 1 |
| `metadata_only` | 1 |
| `drop` | 9,997 |

### 2.3 Findings

1. **Diagnosis is preserved.** The failure event is always observed in full;
   the benchmark asserts that its `input_data` and `exception_type` survive
   observation. This is enforced twice — by the D4 override in the policy and
   by `tests/test_adaptive_evaluation.py` / `tests/integration/test_observation_integration.py`.
2. **Volume is not lost, payload is.** The event count is identical in both
   modes: adaptive observability changes *how much* is retained per event, not
   *which* events exist. The trace skeleton stays queryable.
3. **Storage savings are large but bounded by the skeleton.** The SQLite file
   shrinks 71 %, not 94 %, because dropped events keep their identity columns
   and the index. That is the price of keeping the causal graph intact.
4. **The overhead is negligible.** ~0.6 µs per event on the scoring and
   policy path — far below the cost of the storage write it protects.
5. **The repeat counter is per trace.** Repetition counters are keyed by
   `trace_id` and cleared on `end_trace()`, so one trace can never lower the
   importance of an identical step in another trace.

### 2.4 Recommendation

Keep `full` as the default for local debugging, and use `adaptive` for
long-running or high-volume deployments. The evaluation shows a 71 % storage
reduction for a 5 % CPU cost, with no loss of diagnostic evidence. Before
enabling it in production, review the thresholds against a real workload: the
defaults are tuned so that errors, retries, decisions and final answers always
stay at `FULL`.

---

## 3. Where it lives

| Path | Contents |
|---|---|
| `src/agentlens/observation/importance.py` | `ScoringConfig`, `ImportanceScorer` |
| `src/agentlens/observation/policy.py` | `PolicyConfig`, `ObservationPolicy` |
| `src/agentlens/observation/config.py` | env loading, `build_policy_and_scorer()` |
| `src/agentlens/sdk_collector/collector.py` | optional `scorer` / `policy` injection, fail-open |
| `src/agentlens/web_api.py` | `importance_score` + `observation_level` on every event, `observation_summary` on a run |
| `frontend/app.js` | observation-level badge and importance bar on each event card |
| `tests/test_importance_scoring.py` | scoring unit tests |
| `tests/test_observation_policy.py` | policy unit tests |
| `tests/test_adaptive_evaluation.py` | evaluation invariants |
| `tests/integration/test_observation_integration.py` | end-to-end tests (API, storage, SQLite round-trip) |
| `tests/benchmark_observation.py` | the benchmark above |
