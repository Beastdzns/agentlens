import os
from dataclasses import dataclass
from .importance import ImportanceScorer, ScoringConfig
from .policy import ObservationPolicy, PolicyConfig

@dataclass(frozen=True)
class ObservationConfig:
    policy_config: PolicyConfig
    scoring_config: ScoringConfig

def load_from_env() -> ObservationConfig:
    p_config = PolicyConfig(
        mode=os.getenv("AGENTLENS_OBSERVATION_POLICY", "full"),
        compress_threshold=float(os.getenv("AGENTLENS_COMPRESS_THRESHOLD", "0.60")),
        metadata_threshold=float(os.getenv("AGENTLENS_METADATA_THRESHOLD", "0.35")),
        drop_threshold=float(os.getenv("AGENTLENS_DROP_THRESHOLD", "0.15")),
        payload_max_bytes=int(os.getenv("AGENTLENS_PAYLOAD_MAX_BYTES", "4096"))
    )
    return ObservationConfig(policy_config=p_config, scoring_config=ScoringConfig())

def build_policy_and_scorer(config: ObservationConfig | None = None) -> tuple[ObservationPolicy, ImportanceScorer]:
    cfg = config or load_from_env()
    return ObservationPolicy(cfg.policy_config), ImportanceScorer(cfg.scoring_config)