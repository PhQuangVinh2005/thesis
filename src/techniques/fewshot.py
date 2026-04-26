"""Few-shot prompting — inject golden hallucination-free examples."""

import json
import logging
from typing import List

from ..models.base import BaseLLM
from .base import BaseTechnique

logger = logging.getLogger(__name__)


class FewShotTechnique(BaseTechnique):
    """Few-shot prompting with golden examples.

    Builds its own prompt: instruction + N examples + actual context.
    """

    def __init__(self, examples_file: str, indices: List[int], instruction: str):
        super().__init__(name=f"fewshot_{len(indices)}")
        self.indices = indices
        self.instruction = instruction
        self.examples = self._load_examples(examples_file, indices)
        logger.info(
            f"FewShotTechnique: {len(self.examples)} examples "
            f"(indices={indices}, chars="
            f"{sum(e['text_chars'] + e['summary_chars'] for e in self.examples)})"
        )

    def _load_examples(self, path: str, indices: List[int]) -> list:
        with open(path) as f:
            all_examples = json.load(f)
        selected = []
        for idx in indices:
            if idx < 0 or idx >= len(all_examples):
                raise ValueError(f"Index {idx} out of range (0–{len(all_examples) - 1})")
            selected.append(all_examples[idx])
        return selected

    def _format_examples(self) -> str:
        parts = []
        for i, ex in enumerate(self.examples, 1):
            parts.append(f"Example {i}:\ninput: {ex['text']}\nsummary: {ex['summary']}")
        return "\n\n".join(parts)

    def generate(self, model: BaseLLM, prompt: str, context: str) -> str:
        """Build few-shot prompt and generate. Ignores pipeline's pre-formatted prompt."""
        examples_block = self._format_examples()
        full_prompt = (
            f"{self.instruction}\n\n"
            f"{examples_block}\n\n"
            f"Now summarize this:\n"
            f"input: {context}\n"
            f"summary:"
        )
        result = model.generate(full_prompt)
        self.last_prompt_instruction = (
            f"{self.instruction}\n\n"
            f"{examples_block}\n\n"
            f"Now summarize this:\n"
            f"input: {{context}}\n"
            f"summary:"
        )
        return result
