from dataclasses import dataclass, field
from typing import Mapping
from agentlens.events import AgentEvent, EventStatus, EventType, ObservationLevel
@dataclass(frozen=True)
class ScoringConfig:
    base_by_status: Mapping[EventStatus, float] = field(default_factory=lambda: {
        EventStatus.FAILURE: 0.9, EventStatus.PARTIAL: 0.7,
        EventStatus.PENDING: 0.6, EventStatus.SUCCESS: 0.3
    })
    weight_by_type: Mapping[EventType, float] = field(default_factory=lambda: {
        EventType.ERROR: 1.0, EventType.FINAL_RESPONSE: 0.95, EventType.RETRY: 0.85,
        EventType.DECISION: 0.8, EventType.SUBAGENT_START: 0.7, EventType.SUBAGENT_END: 0.7,
        EventType.TOOL_CALL: 0.55, EventType.LLM_CALL: 0.5, EventType.RETRIEVAL: 0.5,
        EventType.MEMORY_READ: 0.35, EventType.MEMORY_WRITE: 0.35
    })
    parent_bonus: float = 0.10
    latency_high_ms: float = 1000.0
    latency_bonus: float = 0.10
    exception_bonus: float = 0.10
    repetition_penalty: float = 0.15
    repetition_cap: int = 3
    min_score: float = 0.0
    max_score: float = 1.0

class ImportanceScorer:
    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig()

    def score(self, event: AgentEvent, *, repeat_index: int = 0) -> float:
        c = self.config
        s = c.base_by_status.get(event.status, 0.3)
        w = c.weight_by_type.get(event.event_type, 0.5)
        score = s * w
        
        if event.parent_id is None:
            score += c.parent_bonus
        if getattr(event, "latency_ms", 0) > c.latency_high_ms:
            score += c.latency_bonus
        if event.metadata and "exception_type" in event.metadata:
            score += c.exception_bonus
            
        penalty = min(repeat_index, c.repetition_cap) * c.repetition_penalty
        score -= penalty
        return max(c.min_score, min(c.max_score, score))