"""
Technique abstractions for hallucination reduction.

Each technique wraps around the LLM to implement a specific
generation strategy (baseline, self-correction, chain-of-thought, etc.)

Adding a new technique = adding a new file in this directory.
"""

from .base import BaseTechnique
from .baseline import BaselineTechnique
from .fewshot import FewShotTechnique

__all__ = ["BaseTechnique", "BaselineTechnique", "FewShotTechnique"]
