"""BLEU metric for completeness evaluation (P vs L)."""

from typing import Dict, List, Optional

import sacrebleu

from ..base import BaseMetric


class BLEUMetric(BaseMetric):
    """Sentence-level BLEU score (0-100 scale, sacrebleu convention)."""

    def __init__(self):
        super().__init__(name="bleu")

    @property
    def expected_keys(self) -> List[str]:
        return ["bleu_score"]

    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        if reference is None:
            raise ValueError("BLEUMetric requires a reference (labeled summary).")

        result = sacrebleu.sentence_bleu(prediction, [reference])
        return {"bleu_score": result.score}
