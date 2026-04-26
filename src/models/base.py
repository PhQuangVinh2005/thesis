"""Abstract base class for LLM wrappers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLLM(ABC):
    """Unified interface for text generation across backends."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
