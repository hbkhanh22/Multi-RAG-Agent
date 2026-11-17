"""
Base Agent Class - Abstract base for all agents
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging
from datetime import datetime

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, model: str = None, temperature: float = None,
                 max_tokens: int = None):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            model: LLM model (defaults to settings.default_llm_model)
            temperature: Temperature (defaults to settings.llm_temperature)
        """
        self.name = name
        self.model = model or settings.default_llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        
        try:
            self.llm = ChatNVIDIA(
                model=self.model,
                api_key=settings.nvidia_api_key,
                temperature=self.temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                timeout=settings.llm_timeout,
                **({} if settings.llm_enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}),
            )
            logger.info(f"Initialized {self.name} with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for {self.name}: {e}")
            raise
        
        self.created_at = datetime.now()
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run agent - must be implemented by subclasses
        
        Args:
            input_data: Input dictionary with required fields
        
        Returns:
            Output dictionary with results
        """
        pass
    
    def invoke(self, messages: List, system_prompt: str = None, on_token=None) -> str:
        """
        Invoke LLM with messages
        
        Args:
            messages: List of message dicts or Message objects
            system_prompt: Optional system message
        
        Returns:
            LLM response content
        """
        try:
            message_list = []
            
            if system_prompt:
                message_list.append(SystemMessage(content=system_prompt))
            
            for msg in messages:
                if isinstance(msg, str):
                    message_list.append(HumanMessage(content=msg))
                else:
                    message_list.append(msg)
            
            if on_token is None:
                return self.llm.invoke(message_list).content

            # Streaming: forward text chunks as they arrive, return the full text
            parts: List[str] = []
            for chunk in self.llm.stream(message_list):
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    parts.append(text)
                    on_token(text)
            return "".join(parts)
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate input data
        
        Args:
            input_data: Raw input dictionary
        
        Returns:
            Processed input dictionary
        """
        return input_data
    
    def format_context(self, context_items: List[str], title: str = "Context") -> str:
        """
        Format context for LLM prompt
        
        Args:
            context_items: List of context strings
            title: Title for context section
        
        Returns:
            Formatted context string
        """
        if not context_items:
            return ""
        
        formatted = f"\n{title}:\n" + "-" * 50 + "\n"
        for i, item in enumerate(context_items, 1):
            formatted += f"\n[{i}] {item}\n"
        formatted += "-" * 50 + "\n"
        
        return formatted
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "created_at": self.created_at.isoformat(),
        }
