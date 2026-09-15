"""
BaseAgent abstract class for all specialized agents.
"""
import abc
import time
import logging
from typing import Optional
from src.models.schema import ConversationState, AgentState, AgentStatus

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """
    Abstract base class providing timing, lifecycle logging, and blackboard integration.
    """
    def __init__(self, name: str, timeout: float = 1.0):
        self.name = name
        self.timeout = timeout
        self.is_healthy = True

    async def run(self, state: ConversationState) -> ConversationState:
        """
        Execute the agent with performance timing and error containment.
        """
        start_time = time.perf_counter()
        state.update_agent_status(self.name, AgentState.RUNNING, 0.0)
        
        try:
            state = await self.process(state)
            self.is_healthy = True
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            state.update_agent_status(self.name, AgentState.SUCCESS, latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Error in {self.name}: {str(e)}", exc_info=True)
            self.is_healthy = False
            state.update_agent_status(self.name, AgentState.ERROR, latency_ms, str(e))
            # Handle graceful degradation if agent provides fallback
            state = await self.fallback(state, str(e))
            
        return state

    @abc.abstractmethod
    async def process(self, state: ConversationState) -> ConversationState:
        """Core business logic for the agent."""
        pass

    async def fallback(self, state: ConversationState, error_msg: str) -> ConversationState:
        """Default fallback does not alter state further."""
        return state

    def get_status(self) -> AgentStatus:
        """Return basic health status."""
        return AgentStatus(
            agent_name=self.name,
            state=AgentState.SUCCESS if self.is_healthy else AgentState.ERROR
        )
