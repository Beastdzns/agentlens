"""
AgentLens adaptive observability primitives.

This package implements the *scoring* and *policy* halves of adaptive
observability (see ``report.md`` Phase 10):

- :mod:`agentlens.observation.importance` — rule-based importance scoring.
- ``agentlens.observation.policy`` — observation-level routing and payload
  transformation (owned by the Observation Policy work stream).

Nothing in this package is active by default; the collector only applies an
importance scorer / observation policy when one is explicitly provided.
"""

from agentlens.observation.importance import (
    DEFAULT_SCORING_CONFIG,
    ImportanceScorer,
    ScoringConfig,
)

__all__ = [
    "DEFAULT_SCORING_CONFIG",
    "ImportanceScorer",
    "ScoringConfig",
]
