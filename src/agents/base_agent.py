"""
Base Agent Class - Abstract base for all agents
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging
import time

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from src import runtime

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
                **({} if settings.llm_enable_thinking else {"model_kwargs": {"chat_template_kwargs": {"enable_thinking": False}}}),
            )
            logger.info(f"Initialized {self.name} with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for {self.name}: {e}")
            raise

    
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
            
            runtime.check_cancelled()
            started = time.time()
            if on_token is None:
                msg = self.llm.invoke(message_list)
                content = msg.content if isinstance(msg.content, str) else ""
                self._record_usage(message_list, content, getattr(msg, "usage_metadata", None), started)
                return content

            # Streaming: forward text chunks as they arrive, return the full text.
            # Stops early (raises Cancelled) when the user cancels the request.
            parts: List[str] = []
            usage = None
            for chunk in self.llm.stream(message_list):
                if getattr(chunk, "usage_metadata", None):
                    usage = chunk.usage_metadata
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    parts.append(text)
                    on_token(text)
                runtime.check_cancelled()
            content = "".join(parts)
            self._record_usage(message_list, content, usage, started)
            return content
        except runtime.Cancelled:
            logger.info(f"{self.name}: generation cancelled")
            raise
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise
    
    def _record_usage(self, message_list, content: str, usage, started: float) -> None:
        """Token accounting per request (RunContext) and per model (Prometheus)."""
        try:
            estimated = not usage
            if usage:
                prompt = int(usage.get("input_tokens") or 0)
                completion = int(usage.get("output_tokens") or 0)
            else:  # provider did not report usage: ~4 characters per token
                chars = sum(len(m.content) for m in message_list if isinstance(m.content, str))
                prompt, completion = chars // 4, len(content) // 4
            ctx = runtime.current()
            if ctx is not None:
                ctx.add_usage(self.model, prompt, completion, estimated)
            from src.observability import metrics
            metrics.record_llm_call(self.model, self.name, prompt, completion, time.time() - started)
        except Exception as e:  # accounting must never break an answer
            logger.debug(f"usage accounting failed: {e}")
