"""
Evaluation tests for the Phase 11 causal analysis (member N).

Every labelled scenario of ``tests/causal_scenarios.py`` is checked against
the analyzer: the verdict, the root cause, the primary/secondary split, the
affected set, the category and the confidence.

The set is also used to verify that the analyzer degrades *gracefully* when
the payloads are gone: the structure survives, the confidence drops and the
explanation says that no detail was retained — but the analyzer never invents
a cause.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from agentlens.causal import RootCauseAnalyzer, build_graph
from agentlens.causal.explain import NO_DETAIL_PHRASE
from tests.causal_scenarios import (
    SCENARIOS,
    apply_adaptive_observation,
    baseline_first_failure,
    drop_payloads,
    scenario_by_id,
)

pytestmark = pytest.mark.unit


def _analyze(events):
    return RootCauseAnalyzer().analyze(build_graph(events))


# ---------------------------------------------------------------------------
# Per-scenario correctness
# ---------------------------------------------------------------------------


class TestScenarioVerdicts:
    """Every scenario produces its labelled verdict."""

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    def test_verdict_matches_the_label(self, scenario) -> None:
        result = _analyze(scenario.events)
        assert result.verdict == scenario.expected_verdict

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    def test_root_cause_matches_the_label(self, scenario) -> None:
        result = _analyze(scenario.events)
        assert result.root_cause_event_id == scenario.expected_root_cause

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    def test_category_matches_the_label(self, scenario) -> None:
        result = _analyze(scenario.events)
        assert result.category == scenario.expected_category

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    def test_confidence_matches_the_label(self, scenario) -> None:
        result = _analyze(scenario.events)
        assert result.confidence == pytest.approx(scenario.expected_confidence)

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    def test_primary_and_secondary_failures_match_the_label(self, scenario) -> None:
        result = _analyze(scenario.events)
        assert tuple(result.primary_failures) == scenario.expected_primary_failures
        assert tuple(result.secondary_failures) == scenario.expected_secondary_failures

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    def test_affected_events_match_the_label(self, scenario) -> None:
        result = _analyze(scenario.events)
        assert tuple(result.affected_events) == scenario.expected_affected_events

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    def test_explanation_quotes_the_recorded_error(self, scenario) -> None:
        result = _analyze(scenario.events)
        if scenario.expected_error_message:
            assert scenario.expected_error_message in result.explanation


# ---------------------------------------------------------------------------
# Structural properties of the evaluation set
# ---------------------------------------------------------------------------


class TestEvaluationSetProperties:
    """The evaluation set itself has to be sound."""

    def test_every_scenario_has_a_unique_id(self) -> None:
        ids = [scenario.id for scenario in SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_every_scenario_has_events(self) -> None:
        for scenario in SCENARIOS:
            assert scenario.events, scenario.id

    def test_labelled_root_cause_exists_in_the_scenario(self) -> None:
        for scenario in SCENARIOS:
            if scenario.expected_root_cause is None:
                continue
            assert scenario.root_cause_event is not None, scenario.id

    def test_evidence_path_starts_at_the_root_cause(self) -> None:
        for scenario in SCENARIOS:
            result = _analyze(scenario.events)
            if result.root_cause_event_id is None:
                continue
            assert result.evidence_path[0] == result.root_cause_event_id, scenario.id

    def test_evidence_path_only_contains_real_events(self) -> None:
        for scenario in SCENARIOS:
            result = _analyze(scenario.events)
            known = {str(event.event_id) for event in scenario.events}
            for event_id in result.evidence_path:
                assert event_id in known, scenario.id

    def test_analysis_is_deterministic(self) -> None:
        for scenario in SCENARIOS:
            first = _analyze(scenario.events).to_dict()
            second = _analyze(list(reversed(scenario.events))).to_dict()
            assert first == second, scenario.id


# ---------------------------------------------------------------------------
# H3 — the causal graph beats the chronological baseline
# ---------------------------------------------------------------------------


class TestCausalGraphBeatsChronology:
    """
    H3: *causal graphs improve root-cause localization*.

    The baseline is "the first failure, by wall-clock".  On the labelled set
    the graph-based analyzer must localise at least as many root causes, and
    strictly more on the scenarios where the clock misleads.
    """

    def test_graph_based_localisation_is_perfect_on_the_labeled_set(self) -> None:
        correct = 0
        for scenario in SCENARIOS:
            result = _analyze(scenario.events)
            if result.root_cause_event_id == scenario.expected_root_cause:
                correct += 1
        assert correct == len(SCENARIOS)

    def test_baseline_mislocalises_the_clock_misleading_scenarios(self) -> None:
        """The baseline is wrong exactly where the clock lies."""
        wrong = [
            scenario.id
            for scenario in SCENARIOS
            if baseline_first_failure(scenario.events) != scenario.expected_root_cause
        ]
        assert "skewed_clock" in wrong
        assert "retry_recovered" in wrong

    def test_graph_based_localisation_beats_the_baseline(self) -> None:
        graph_correct = 0
        baseline_correct = 0
        for scenario in SCENARIOS:
            expected = scenario.expected_root_cause
            if _analyze(scenario.events).root_cause_event_id == expected:
                graph_correct += 1
            if baseline_first_failure(scenario.events) == expected:
                baseline_correct += 1

        assert graph_correct > baseline_correct, (
            f"graph {graph_correct}/{len(SCENARIOS)} vs "
            f"baseline {baseline_correct}/{len(SCENARIOS)}"
        )


# ---------------------------------------------------------------------------
# Robustness — degraded payloads
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """D5: the diagnosis degrades, it never breaks and never invents."""

    @pytest.mark.parametrize(
        "scenario",
        [s for s in SCENARIOS if s.survives_payload_drop],
        ids=lambda s: s.id,
    )
    def test_root_cause_survives_adaptive_observation(self, scenario) -> None:
        """Adaptive mode keeps the structure, so the answer stays the same."""
        degraded = apply_adaptive_observation(scenario.events)
        result = _analyze(degraded)
        assert result.root_cause_event_id == scenario.expected_root_cause

    @pytest.mark.parametrize(
        "scenario",
        [s for s in SCENARIOS if s.survives_payload_drop],
        ids=lambda s: s.id,
    )
    def test_confidence_is_capped_when_the_payload_is_gone(self, scenario) -> None:
        degraded = drop_payloads(scenario.events)
        result = _analyze(degraded)
        assert result.confidence <= 0.5

    def test_explanation_reports_the_missing_detail(self) -> None:
        scenario = scenario_by_id("tool_error")
        result = _analyze(drop_payloads(scenario.events))
        assert NO_DETAIL_PHRASE in result.explanation.lower()

    def test_no_cause_is_invented_for_a_dropped_failure(self) -> None:
        scenario = scenario_by_id("tool_error")
        result = _analyze(drop_payloads(scenario.events))
        assert "search backend unavailable" not in result.explanation
        assert result.root_cause_event_id == scenario.expected_root_cause

    def test_scenario_losing_its_evidence_degrades_to_the_failing_step(self) -> None:
        """
        The bad-upstream scenario *is* its payload: without it the analyzer
        can only report the step that actually failed.
        """
        scenario = scenario_by_id("bad_upstream")
        result = _analyze(apply_adaptive_observation(scenario.events))
        assert result.verdict == "root_cause"
        assert result.root_cause_event_id == str(scenario.events[1].event_id)
        assert result.confidence <= 0.5

    def test_retry_link_survives_adaptive_observation(self) -> None:
        """
        The retry event is kept at FULL by the policy, so the ``retry_of``
        link — and with it the ``recovered`` verdict — survives.
        """
        scenario = scenario_by_id("retry_recovered")
        result = _analyze(apply_adaptive_observation(scenario.events))
        assert result.verdict == "recovered"
        assert result.root_cause_event_id is None

    def test_retry_link_is_lost_when_everything_is_dropped(self) -> None:
        """The hostile case: without the metadata the retry is invisible."""
        scenario = scenario_by_id("retry_recovered")
        result = _analyze(drop_payloads(scenario.events))
        assert result.verdict == "root_cause"
        assert result.root_cause_event_id == str(scenario.events[0].event_id)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_analysis_of_a_thousand_events_stays_under_100_ms() -> None:
    """NFR1: the analyzer must stay cheap on large traces."""
    import time

    scenario = scenario_by_id("cascade")
    events = list(scenario.events)
    root = events[0]
    for index in range(1, 1000):
        events.append(
            root.model_copy(
                update={
                    "event_id": UUID(int=index),
                    "parent_id": root.event_id,
                }
            )
        )

    start = time.perf_counter()
    result = _analyze(events)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result.root_cause_event_id == str(root.event_id)
    assert elapsed_ms < 100, f"analysis took {elapsed_ms:.1f} ms"
