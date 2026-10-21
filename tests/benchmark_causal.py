"""
Phase 11 benchmark: causal-graph localization vs. chronological baseline.

Run it directly::

    ./venv/bin/python tests/benchmark_causal.py

It measures four things on the labelled scenario set of
``tests/causal_scenarios.py``:

- **H3 — localization**  — how often the causal analyzer names the labelled
  root cause, compared with the naive baseline "the first failure, by
  wall-clock";
- **latency**           — the cost of building the graph and diagnosing it,
  per trace size;
- **robustness**        — what survives when adaptive observation has thrown
  the payloads away;
- **explanation quality**— whether the grounded explanation quotes the
  recorded error message instead of inventing one.

The numbers printed here are the ones quoted in
``docs/phase11-causal-analysis.md``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

# Allow running the file directly (``python tests/benchmark_causal.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentlens.causal import RootCauseAnalyzer, build_graph  # noqa: E402
from agentlens.causal.explain import NO_DETAIL_PHRASE
from agentlens.events import AgentEvent, EventStatus, EventType  # noqa: E402
from tests.causal_scenarios import (  # noqa: E402
    SCENARIOS,
    apply_adaptive_observation,
    baseline_first_failure,
    drop_payloads,
)

#: Sizes of the synthetic traces used for the latency measurement.
TRACE_SIZES = (100, 1_000, 10_000)


@dataclass(frozen=True)
class LocalizationRow:
    """One row of the H3 comparison table."""

    scenario: str
    expected: Optional[str]
    graph_answer: Optional[str]
    baseline_answer: Optional[str]

    @property
    def graph_hit(self) -> bool:
        return self.graph_answer == self.expected

    @property
    def baseline_hit(self) -> bool:
        return self.baseline_answer == self.expected


def _analyze(events):
    return RootCauseAnalyzer().analyze(build_graph(events))


def measure_localization() -> list[LocalizationRow]:
    """Compare the graph-based analyzer with the chronological baseline."""
    rows: list[LocalizationRow] = []
    for scenario in SCENARIOS:
        result = _analyze(scenario.events)
        rows.append(
            LocalizationRow(
                scenario=scenario.id,
                expected=scenario.expected_root_cause,
                graph_answer=result.root_cause_event_id,
                baseline_answer=baseline_first_failure(scenario.events),
            )
        )
    return rows


def measure_robustness() -> list[dict[str, Any]]:
    """How much of the diagnosis survives adaptive observation."""
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        adaptive = _analyze(apply_adaptive_observation(scenario.events))
        hostile = _analyze(drop_payloads(scenario.events))
        rows.append(
            {
                "scenario": scenario.id,
                "expected": scenario.expected_root_cause,
                "adaptive": adaptive.root_cause_event_id,
                "adaptive_confidence": adaptive.confidence,
                "hostile": hostile.root_cause_event_id,
                "hostile_confidence": hostile.confidence,
                "survives": scenario.survives_payload_drop,
            }
        )
    return rows


def measure_explanations() -> list[dict[str, Any]]:
    """Whether the explanation is grounded in the recorded evidence."""
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        result = _analyze(scenario.events)
        degraded = _analyze(drop_payloads(scenario.events))
        message = scenario.expected_error_message
        rows.append(
            {
                "scenario": scenario.id,
                "quotes_recorded_error": bool(message) and message in result.explanation,
                "degraded_says_no_detail": NO_DETAIL_PHRASE in degraded.explanation.lower(),
                "length": len(result.explanation),
            }
        )
    return rows


def _synthetic_trace(event_count: int) -> list[AgentEvent]:
    """A flat trace of *event_count* events that ends in a failure."""
    from datetime import datetime, timezone

    events: list[AgentEvent] = []
    trace_id = UUID("33333333-3333-4333-8333-333333333333")
    root_id = UUID("44444444-4444-4444-8444-444444444444")
    for index in range(event_count):
        is_last = index == event_count - 1
        events.append(
            AgentEvent(
                event_id=UUID(int=index + 1),
                trace_id=trace_id,
                parent_id=None if index == 0 else root_id,
                agent_id="benchmark-agent",
                event_type=EventType.ERROR if is_last else EventType.LLM_CALL,
                status=EventStatus.FAILURE if is_last else EventStatus.SUCCESS,
                timestamp=datetime.now(timezone.utc),
                input_data={"query": f"step {index}"},
                output_data={"error": "boom"} if is_last else {"result": "ok"},
                latency_ms=1.0,
            )
        )
    return events


def measure_latency() -> list[dict[str, Any]]:
    """Graph construction + analysis, per trace size."""
    rows: list[dict[str, Any]] = []
    for size in TRACE_SIZES:
        events = _synthetic_trace(size)

        start = time.perf_counter()
        graph = build_graph(events)
        build_ms = (time.perf_counter() - start) * 1000

        samples: list[float] = []
        for _ in range(5):
            start = time.perf_counter()
            RootCauseAnalyzer().analyze(graph)
            samples.append((time.perf_counter() - start) * 1000)

        samples.sort()
        rows.append(
            {
                "events": size,
                "build_ms": build_ms,
                "analyze_mean_ms": sum(samples) / len(samples),
                "analyze_p95_ms": samples[int(len(samples) * 0.95) - 1],
            }
        )
    return rows


def main() -> None:
    print("Phase 11 benchmark — causal analysis")
    print("=" * 78)

    # -- H3: localization ---------------------------------------------------
    rows = measure_localization()
    graph_hits = sum(1 for row in rows if row.graph_hit)
    baseline_hits = sum(1 for row in rows if row.baseline_hit)

    print()
    print("H3 — root-cause localization on the labelled set")
    print(f"{'scenario':<20}{'graph':>10}{'baseline':>12}{'expected':>12}")
    print("-" * 54)
    for row in rows:
        graph_mark = "hit" if row.graph_hit else "MISS"
        base_mark = "hit" if row.baseline_hit else "MISS"
        print(f"{row.scenario:<20}{graph_mark:>10}{base_mark:>12}{'yes' if row.expected else 'none':>12}")
    print("-" * 54)
    print(
        f"{'accuracy':<20}{f'{graph_hits}/{len(rows)}':>10}"
        f"{f'{baseline_hits}/{len(rows)}':>12}"
    )
    print(
        f"{'accuracy %':<20}"
        f"{graph_hits / len(rows) * 100:>9.1f}%"
        f"{baseline_hits / len(rows) * 100:>11.1f}%"
    )

    # -- robustness --------------------------------------------------------
    robustness = measure_robustness()
    print()
    print("Robustness — what survives when the payloads are gone")
    print(
        f"{'scenario':<20}{'full':>10}{'adaptive':>12}{'hostile':>12}{'adaptive conf':>16}"
    )
    print("-" * 70)
    for row in robustness:
        adaptive_mark = "hit" if row["adaptive"] == row["expected"] else "MISS"
        hostile_mark = "hit" if row["hostile"] == row["expected"] else "MISS"
        print(
            f"{row['scenario']:<20}{'hit':>10}{adaptive_mark:>12}{hostile_mark:>12}"
            f"{row['adaptive_confidence']:>16.2f}"
        )
    survived = sum(1 for row in robustness if row["adaptive"] == row["expected"])
    print("-" * 70)
    print(f"adaptive mode keeps the root cause in {survived}/{len(robustness)} scenarios")

    # -- explanation quality ----------------------------------------------
    explanations = measure_explanations()
    quoted = sum(1 for row in explanations if row["quotes_recorded_error"])
    honest = sum(1 for row in explanations if row["degraded_says_no_detail"])
    print()
    print("Explanation quality")
    print(f"  quotes the recorded error message : {quoted}/{len(explanations)}")
    print(f"  says 'no detail was retained' when degraded: {honest}/{len(explanations)}")
    print(f"  mean explanation length            : "
          f"{sum(row['length'] for row in explanations) / len(explanations):.0f} characters")

    # -- latency -----------------------------------------------------------
    latency = measure_latency()
    print()
    print("Latency")
    print(f"{'events':>10}{'build (ms)':>14}{'analyze mean':>16}{'analyze p95':>16}")
    print("-" * 56)
    for row in latency:
        print(
            f"{row['events']:>10,}{row['build_ms']:>14.2f}"
            f"{row['analyze_mean_ms']:>16.2f}{row['analyze_p95_ms']:>16.2f}"
        )

    print()
    print(
        json.dumps(
            {
                "h3_graph_accuracy": graph_hits / len(rows),
                "h3_baseline_accuracy": baseline_hits / len(rows),
                "adaptive_survival": survived / len(robustness),
                "latency": latency,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
