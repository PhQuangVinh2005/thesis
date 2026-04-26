"""
Baseline technique — direct single-pass generation.

No hallucination reduction applied. This is the control/baseline
that all other techniques will be compared against.
"""

from ..models.base import BaseLLM
from .base import BaseTechnique


class BaselineTechnique(BaseTechnique):
    """Baseline: pass prompt directly to model, return response."""

    def __init__(self):
        super().__init__(name="baseline")

    def generate(self, model: BaseLLM, prompt: str, context: str) -> str:
        return model.generate(prompt)
