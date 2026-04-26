"""ROUGE-1/2/L metric for completeness evaluation (P vs L)."""

from typing import Dict, List, Optional

from rouge_score import rouge_scorer

from ..base import BaseMetric


class ROUGEMetric(BaseMetric):
    """ROUGE-1, ROUGE-2, ROUGE-L. Returns 9 scores (P/R/F × 3)."""

    def __init__(self, use_stemmer: bool = True):
        super().__init__(name="rouge")
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
        )

    @property
    def expected_keys(self) -> List[str]:
        return [
            "rouge1_precision", "rouge1_recall", "rouge1_f",
            "rouge2_precision", "rouge2_recall", "rouge2_f",
            "rougeL_precision", "rougeL_recall", "rougeL_f",
        ]

    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        if reference is None:
            raise ValueError("ROUGEMetric requires a reference (labeled summary).")

        scores = self.scorer.score(reference, prediction)
        return {
            "rouge1_precision": scores["rouge1"].precision,
            "rouge1_recall": scores["rouge1"].recall,
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_precision": scores["rouge2"].precision,
            "rouge2_recall": scores["rouge2"].recall,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_precision": scores["rougeL"].precision,
            "rougeL_recall": scores["rougeL"].recall,
            "rougeL_f": scores["rougeL"].fmeasure,
        }
