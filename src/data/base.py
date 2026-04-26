"""
Abstract base class for data loaders.

Extend this class to add new datasets (MIMIC-IV-BHC, custom, etc.)
Designed for reusability: subclass for new datasets, swap in pipelines.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import random

from .schema import EvalSample


class BaseDataLoader(ABC):
    """
    Abstract data loader interface.

    Subclasses must implement:
    - load(): Load raw data and return list of EvalSamples
    - preprocess(): Clean/normalize raw clinical text

    Provides:
    - sample(): Reproducible stratified random sampling
    - __getitem__, __iter__, __len__: Standard iteration support
    """

    def __init__(self, data_dir: str, **kwargs):
        self.data_dir = data_dir
        self.config = kwargs
        self._samples: List[EvalSample] = []

    @abstractmethod
    def load(self, **kwargs) -> List[EvalSample]:
        """Load and return all evaluation samples."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, raw_text: str) -> str:
        """Clean and normalize raw clinical text."""
        raise NotImplementedError

    def sample(
        self,
        n: Optional[int] = None,
        seed: int = 42,
    ) -> List[EvalSample]:
        """
        Random sample from loaded data with reproducibility.

        Args:
            n: Number of samples. None = return all.
            seed: Random seed for reproducibility.

        Returns:
            List of sampled EvalSamples.
        """
        if not self._samples:
            raise RuntimeError("No data loaded. Call load() first.")

        if n is None or n >= len(self._samples):
            return list(self._samples)

        rng = random.Random(seed)
        return rng.sample(self._samples, n)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]

    def __iter__(self):
        return iter(self._samples)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_dir='{self.data_dir}', n={len(self)})"
