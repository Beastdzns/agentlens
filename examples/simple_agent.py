"""
Simple example AI agent for testing AgentLens.

PHASE 1 PLACEHOLDER:
This module contains only a skeleton for future development.

Future implementation:
- SimpleAgent: Basic agent that performs LLM calls, tool calls, retrieval
- Integrated with AgentLens SDK
- Used for development, testing, and benchmarking
- Supports both success and failure scenarios for diagnosis evaluation

Example usage (future):

    agent = SimpleAgent()
    collector = EventCollector(storage_backend=...)
    
    async with collector.trace("request-1") as trace:
        result = await agent.run(
            user_query="Find information about X",
            max_steps=10,
        )
        
    # AgentLens has captured structured execution events
    # Failure diagnosis and visualization now available in dashboard
"""


class SimpleAgent:
    """
    PLACEHOLDER: A simple demonstrative AI agent.
    
    PHASE 1 will implement:
    - Initialize with an LLM client (e.g., OpenAI, Anthropic)
    - Plan step based on user query
    - Execute tool calls
    - Process results
    - Emit structured events to AgentLens
    
    Operations that should generate events:
    - LLM call to create plan
    - Search tool invocation
    - Result retrieval
    - Decision making
    - Retries on failure
    - Final response generation
    """
    
    def __init__(self) -> None:
        """Initialize the simple agent."""
        pass
    
    async def run(
        self,
        user_query: str,
        max_steps: int = 10,
    ) -> str:
        """
        Run the agent on a user query.
        
        Args:
            user_query: User's natural language request
            max_steps: Maximum planning steps to prevent infinite loops
            
        Returns:
            Final response to the user
        """
        raise NotImplementedError("Phase 1 implementation required")
