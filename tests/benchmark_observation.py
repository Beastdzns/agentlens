import time
import json
from uuid import uuid4
from datetime import datetime, timezone
from agentlens.events import AgentEvent, EventStatus, EventType
from agentlens.observation import ImportanceScorer, ObservationPolicy, PolicyConfig

def generate_synthetic_trace(event_count: int = 1000):
    events = []
    t_id = str(uuid4())
    for i in range(event_count):
        is_failure = (i == event_count - 1)
        events.append(AgentEvent(
            event_id=str(uuid4()),
            trace_id=t_id,
            parent_id=None if i == 0 else events[0].event_id,
            agent_id="benchmark-agent",
            event_type=EventType.ERROR if is_failure else EventType.LLM_CALL,
            status=EventStatus.FAILURE if is_failure else EventStatus.SUCCESS,
            timestamp=datetime.now(timezone.utc),
            input_data={"context": "A" * 500, "query": f"iter {i}"},
            output_data={"raw": "B" * 500, "parsed": {"status": "ok"}},
            latency_ms=100.0,
            metadata={"exception_type": "TimeoutError"} if is_failure else {}
        ))
    return events

def run_benchmark(mode: str, events: list[AgentEvent]):
    scorer = ImportanceScorer()
    policy = ObservationPolicy(PolicyConfig(mode=mode))
    
    start_time = time.perf_counter()
    transformed_events = []
    total_bytes = 0
    
    for i, e in enumerate(events):
        score = scorer.score(e, repeat_index=i if e.event_type == EventType.LLM_CALL else 0)
        level = policy.decide(score, e)
        final_e = policy.transform(e, level)
        transformed_events.append(final_e)
        total_bytes += len(json.dumps(final_e.model_dump(), default=str).encode("utf-8"))
        
    duration = time.perf_counter() - start_time
    
    # Diagnosis check
    failure_event = transformed_events[-1]
    diagnosis_preserved = (
        failure_event.status == EventStatus.FAILURE and
        len(str(failure_event.input_data)) > 100 and 
        "exception_type" in failure_event.metadata
    )
    
    print(f"--- Mode: {mode.upper()} ---")
    print(f"Time to process {len(events)} events: {duration:.4f}s")
    print(f"Total payload size: {total_bytes / 1024:.2f} KB")
    print(f"Diagnosis accuracy preserved: {diagnosis_preserved}\n")

if __name__ == "__main__":
    print("Generating 10,000 synthetic events...")
    events = generate_synthetic_trace(10000)
    
    run_benchmark("full", events)
    run_benchmark("adaptive", events)