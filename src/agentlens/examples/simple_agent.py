import os
import time

from dotenv import load_dotenv
from typing import Optional

from google import genai

from agentlens.collector import EventCollector
from agentlens.events import EventStatus, Trace

load_dotenv()


class SimpleAgent:
    """
    AI agent powered by Google's Gemini model with AgentLens instrumentation.
    
    Automatically captures all LLM calls as structured events for observability.
    
    Attributes:
        client: Gemini Client instance.
        model_name: Name of the Gemini model to execute.
        collector: EventCollector for capturing events.
    """
    
    def __init__(self, api_key: str, model_name: Optional[str] = None) -> None:
        """
        Initialize the agent with Gemini API.
        
        Args:
            api_key: Google Generative AI API key.
            model_name: Gemini model to use.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.collector = EventCollector()
    
    async def run(
        self,
        query: str,
        agent_id: str = "simple-agent",
    ) -> dict:
        """
        Run the agent on a query with automatic event capture.
        
        Args:
            query: User query to process.
            agent_id: Identifier for the agent.
            
        Returns:
            Dictionary with response, trace_id, event count, and latency.
            
        Raises:
            RuntimeError: If Gemini API call fails.
        """
        trace = self.collector.start_trace(agent_id=agent_id)
        
        try:
            # Record timing
            start_time = time.time()
            
            # Call Gemini via the async interface (.aio.models) with modern parameters
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=query,
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Record the event
            self.collector.record_llm_call(
                agent_id=agent_id,
                prompt=query,
                response=response.text,
                latency_ms=latency_ms,
                model=self.model_name,
                status=EventStatus.SUCCESS,
            )
            
            # Finalize trace
            final_trace = self.collector.end_trace(EventStatus.SUCCESS)
            
            return {
                "response": response.text,
                "trace_id": str(final_trace.trace_id),
                "events": final_trace.event_count,
                "latency_ms": final_trace.total_latency_ms,
                "trace": final_trace,
            }
        
        except Exception as e:
            # Record failure and re-raise
            self.collector.end_trace(EventStatus.FAILURE)
            raise RuntimeError(f"Agent failed: {str(e)}") from e
    
    def get_current_trace(self) -> Optional[Trace]:
        """
        Get the currently active trace.
        
        Returns:
            Current Trace or None if no trace is active.
        """
        return self.collector.get_current_trace()