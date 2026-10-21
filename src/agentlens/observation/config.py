"""Environment-driven configuration and factory for observation policy.

Env vars:
    AGENTLENS_OBSERVATION_POLICY  full | adaptive   (default: full)
    AGENTLENS_COMPRESS_THRESHOLD  float 0-1         (default: 0.60)
    AGENTLENS_METADATA_THRESHOLD  float 0-1         (default: 0.35)
    AGENTLENS_DROP_THRESHOLD      float 0-1         (default: 0.15)
    AGENTLENS_PAYLOAD_MAX_BYTES   int               (default: 4096)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agentlens.observation.importance import ImportanceScorer, ScoringConfig
from agentlens.observation.policy import ObservationPolicy, PolicyConfig


@dataclass(frozen=True)
class ObservationConfig:
    """Merged env-driven config for scoring + policy."""

    mode: str = "full"
    compress_threshold: float = 0.60
    metadata_threshold: float = 0.35
    drop_threshold: float = 0.15
    payload_max_bytes: int = 4096


def load_from_env() -> ObservationConfig:
    """Read observation config from environment variables."""
    return ObservationConfig(
        mode=os.getenv("AGENTLENS_OBSERVATION_POLICY", "full").lower(),
        compress_threshold=float(os.getenv("AGENTLENS_COMPRESS_THRESHOLD", "0.60")),
        metadata_threshold=float(os.getenv("AGENTLENS_METADATA_THRESHOLD", "0.35")),
        drop_threshold=float(os.getenv("AGENTLENS_DROP_THRESHOLD", "0.15")),
        payload_max_bytes=int(os.getenv("AGENTLENS_PAYLOAD_MAX_BYTES", "4096")),
    )


def build_policy_and_scorer(
    cfg: ObservationConfig | None = None,
) -> tuple[ImportanceScorer, ObservationPolicy]:
    """Factory: create an ImportanceScorer + ObservationPolicy from config.

    D6: when mode is 'full' (the default), the policy is a no-op.
    D8: adaptive is opt-in via env var.

    Returns:
        (scorer, policy) ready for injection into EventCollector.
    """
    if cfg is None:
        cfg = load_from_env()

    scorer = ImportanceScorer(ScoringConfig())
    policy = ObservationPolicy(
        PolicyConfig(
            mode=cfg.mode,
            compress_threshold=cfg.compress_threshold,
            metadata_threshold=cfg.metadata_threshold,
            drop_threshold=cfg.drop_threshold,
            payload_max_bytes=cfg.payload_max_bytes,
        )
    )
    return scorer, policy
