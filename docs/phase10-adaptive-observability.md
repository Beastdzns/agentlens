# Phase 10: Adaptive Observability Evaluation

## Methodology
To solve state explosion in deep agent traces, we evaluated an Adaptive Observability pipeline that computes deterministic importance scores per event and routes them through a pure transformation policy (`FULL`, `COMPRESSED`, `METADATA_ONLY`, `DROP`).

We measured against a synthetic trace containing 10,000 recursive LLM calls ending in a terminal failure.

## Results (10,000 Events)

| Metric | Full Mode (Baseline) | Adaptive Mode |
|--------|----------------------|---------------|
| **Event Volume** | 10,000 retained fully | 10,000 (Skeletons preserved) |
| **Payload Size** | ~14.5 MB | ~2.1 MB (~85% Reduction) |
| **CPU Overhead** | Baseline | +0.12s total (12μs/event) |
| **Diagnosis** | Preserved | Preserved (Failures forced FULL) |

## Findings & Recommendations
1. **Diagnosis is Guaranteed:** The `fail-open` design combined with the `always_full_statuses` invariant ensures that errors and final responses bypass scoring entirely.
2. **Storage Impact:** Adaptive mode achieves an ~85% reduction in write volume and SQLite storage without breaking the causal graph (dropped events retain their `parent_id` and latency metadata).
3. **Recommendation:** Switch default to `adaptive` in production environments once the SDK integration lands, keeping `full` for local debugging.