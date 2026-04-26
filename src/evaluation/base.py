"""Abstract base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseMetric(ABC):
    """Strategy pattern metric — compute() returns Dict[str, float]."""

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def expected_keys(self) -> List[str]:
        """Keys that compute() returns. Used for error fallback."""
        ...

    @abstractmethod
    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
