from dataclasses import dataclass, field
from typing import Any
from agentlens.events import AgentEvent, EventStatus, EventType, ObservationLevel

@dataclass(frozen=True)
class PolicyConfig:
    mode: str = "full"
    compress_threshold: float = 0.60
    metadata_threshold: float = 0.35
    drop_threshold: float = 0.15
    payload_max_bytes: int = 4096
    always_full_types: frozenset = field(default_factory=lambda: frozenset({EventType.ERROR, EventType.RETRY, EventType.FINAL_RESPONSE}))
    always_full_statuses: frozenset = field(default_factory=lambda: frozenset({EventStatus.FAILURE}))
    drop_retain_skeleton: bool = True

class ObservationPolicy:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def decide(self, importance_score: float, event: AgentEvent) -> ObservationLevel:
        if self.config.mode == "full":
            return ObservationLevel.FULL
        if event.event_type in self.config.always_full_types or event.status in self.config.always_full_statuses:
            return ObservationLevel.FULL
            
        if importance_score >= self.config.compress_threshold:
            return ObservationLevel.FULL
        if importance_score >= self.config.metadata_threshold:
            return ObservationLevel.COMPRESSED
        if importance_score >= self.config.drop_threshold:
            return ObservationLevel.METADATA_ONLY
        return ObservationLevel.DROP

    def transform(self, event: AgentEvent, level: ObservationLevel) -> AgentEvent:
        if level == ObservationLevel.FULL:
            return event
            
        updates: dict[str, Any] = {"observation_level": level}
        
        if level == ObservationLevel.COMPRESSED:
            def truncate(d: Any) -> Any:
                if not isinstance(d, dict): return d
                return {k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in d.items()}
            updates["input_data"] = truncate(event.input_data)
            updates["output_data"] = truncate(event.output_data)

        elif level == ObservationLevel.METADATA_ONLY:
            updates["input_data"] = {}
            updates["output_data"] = {}

        elif level == ObservationLevel.DROP:
            if self.config.drop_retain_skeleton:
                updates["input_data"] = {}
                updates["output_data"] = {}
                updates["metadata"] = {**(event.metadata or {}), "_dropped": True}
        
        if hasattr(event, "model_copy"):
            return event.model_copy(update=updates)
        return event.copy(update=updates)