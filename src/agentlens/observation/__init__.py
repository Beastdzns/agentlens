"""
AgentLens adaptive observability.

This package implements the *scoring* and *policy* halves of adaptive
observability (see ``report.md`` Phase 10):

- :mod:`agentlens.observation.importance` — rule-based importance scoring.
- :mod:`agentlens.observation.policy` — observation-level routing and payload
  transformation.
- :mod:`agentlens.observation.config` — environment-driven configuration and
  the ``build_policy_and_scorer`` factory.

Nothing in this package is active by default; the collector only applies an
importance scorer / observation policy when one is explicitly provided.
"""

from agentlens.observation.config import (
    ObservationConfig,
    build_policy_and_scorer,
    load_from_env,
)
from agentlens.observation.importance import (
    DEFAULT_SCORING_CONFIG,
    ImportanceScorer,
    ScoringConfig,
)
from agentlens.observation.policy import ObservationPolicy, PolicyConfig

__all__ = [
    "DEFAULT_SCORING_CONFIG",
    "ImportanceScorer",
    "ObservationConfig",
    "ObservationPolicy",
    "PolicyConfig",
    "ScoringConfig",
    "build_policy_and_scorer",
    "load_from_env",
]
