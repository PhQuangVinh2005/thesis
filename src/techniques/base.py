"""Abstract base class for hallucination reduction techniques."""

from abc import ABC, abstractmethod

from ..models.base import BaseLLM


class BaseTechnique(ABC):
    """Extension point for hallucination reduction.

    Subclasses: BaselineTechnique, FewShotTechnique, (future: CoVe, etc.)
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, model: BaseLLM, prompt: str, context: str) -> str:
        """Generate a summary using this technique.

        Args:
            model: LLM instance
            prompt: Formatted prompt (instruction + context)
            context: Raw medical record (for techniques that re-reference source)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
