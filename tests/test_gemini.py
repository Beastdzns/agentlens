import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, "/home/shiro/Desktop/agentlens/src")

from agentlens.examples.simple_agent import SimpleAgent
from agentlens.events import EventType, EventStatus

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not set")
        print("   Run: export GEMINI_API_KEY='your-key'")
        return
    
    print("🚀 Testing Gemini integration with AgentLens...\n")
    
    agent = SimpleAgent(api_key=api_key)
    
    try:
        # Simple query
        result = await agent.run(
            query="Explain quantum computing in one sentence",
            agent_id="test-agent"
        )
        
        print("✅ SUCCESS!\n")
        print(f"Response: {result['response'][:200]}...")
        print(f"\nTrace Metadata:")
        print(f"  • Trace ID: {result['trace_id']}")
        print(f"  • Events captured: {result['events']}")
        print(f"  • Latency: {result['latency_ms']:.2f}ms")
        
        # Inspect the trace
        trace = result['trace']
        llm_events = trace.get_events_by_type(EventType.LLM_CALL)
        print(f"\nEvent Details:")
        for event in llm_events:
            print(f"  • Type: {event.event_type.value}")
            print(f"  • Status: {event.status.value}")
            print(f"  • Latency: {event.latency_ms:.2f}ms")
            print(f"  • Input: {event.input_data['model']}")
    
    except Exception as e:
        print(f"❌ FAILED: {str(e)}\n")
        if "API key" in str(e):
            print("   Check your GEMINI_API_KEY is correct")
        elif "429" in str(e):
            print("   Rate limit hit - try again in a minute")
        elif "permission denied" in str(e).lower():
            print("   API key doesn't have permission - regenerate it")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())