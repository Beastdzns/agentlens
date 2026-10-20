import pytest
from datetime import datetime, timezone
from uuid import uuid4
from agentlens.events import AgentEvent, EventStatus, EventType, ObservationLevel
from agentlens.observation import ImportanceScorer, ObservationPolicy, PolicyConfig

@pytest.fixture
def base_event():
    return AgentEvent(
        event_id=str(uuid4()),
        trace_id=str(uuid4()),
        parent_id=str(uuid4()),
        agent_id="test-agent",
        event_type=EventType.LLM_CALL,
        status=EventStatus.SUCCESS,
        timestamp=datetime.now(timezone.utc),
        input_data={"prompt": "test " * 100},
        output_data={"response": "result " * 100},
        latency_ms=150.0,
        metadata={}
    )

def test_failure_is_always_full(base_event):
    policy = ObservationPolicy(PolicyConfig(mode="adaptive"))
    event = base_event.model_copy(update={"status": EventStatus.FAILURE, "event_type": EventType.LLM_CALL})
    
    level = policy.decide(0.0, event)  # Force score to 0
    assert level == ObservationLevel.FULL
    
    transformed = policy.transform(event, level)
    assert transformed.input_data == event.input_data
    assert transformed.output_data == event.output_data

def test_drop_retains_skeleton(base_event):
    policy = ObservationPolicy(PolicyConfig(mode="adaptive", drop_retain_skeleton=True))
    level = ObservationLevel.DROP
    
    transformed = policy.transform(base_event, level)
    
    # Structure remains
    assert transformed.event_id == base_event.event_id
    assert transformed.trace_id == base_event.trace_id
    assert transformed.parent_id == base_event.parent_id
    assert transformed.status == base_event.status
    
    # Payloads wiped
    assert transformed.input_data == {}
    assert transformed.output_data == {}
    assert transformed.metadata["_dropped"] is True

def test_repetition_penalty_drops_score(base_event):
    scorer = ImportanceScorer()
    # Boost the initial score so the penalty doesn't hit the 0.0 floor!
    high_event = base_event.model_copy(update={"status": EventStatus.PARTIAL, "event_type": EventType.DECISION})
    score_first = scorer.score(high_event, repeat_index=0)
    score_third = scorer.score(high_event, repeat_index=3)
    
    assert score_third < score_first
    assert score_first - score_third == pytest.approx(0.45) # 3 * 0.15

def test_full_mode_no_op(base_event):
    policy = ObservationPolicy(PolicyConfig(mode="full"))
    level = policy.decide(0.0, base_event)
    assert level == ObservationLevel.FULL
    transformed = policy.transform(base_event, level)
    assert transformed.input_data == base_event.input_data