"""BERTScore metric for semantic similarity evaluation (P vs L)."""

import logging
from typing import Dict, List, Optional

from ..base import BaseMetric

logger = logging.getLogger(__name__)

# ── Fix: OverflowError with transformers 5.x ──────────────────────────
# DeBERTa models don't define model_max_length → defaults to ~10^30
# → Rust tokenizer overflows on enable_truncation().
# Monkey-patch AutoTokenizer.from_pretrained to cap the value.
_PATCHED = False

def _patch_tokenizer_overflow():
    """One-time patch: cap model_max_length for all tokenizers."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from transformers import AutoTokenizer
        _original_from_pretrained = AutoTokenizer.from_pretrained.__func__

        @classmethod
        def _patched_from_pretrained(cls, *args, **kwargs):
            tokenizer = _original_from_pretrained(cls, *args, **kwargs)
            if tokenizer.model_max_length > 1_000_000:
                tokenizer.model_max_length = 512
            return tokenizer

        AutoTokenizer.from_pretrained = _patched_from_pretrained
        logger.debug("Patched AutoTokenizer.from_pretrained (model_max_length cap)")
    except Exception as e:
        logger.warning(f"Could not patch tokenizer: {e}")
    _PATCHED = True
# ───────────────────────────────────────────────────────────────────────


class BERTScoreMetric(BaseMetric):
    """BERTScore — semantic similarity using contextual embeddings.

    Uses BERTScorer class (not bert_score.score function) to cache the
    model/tokenizer across calls — avoids re-downloading from HF Hub
    on every sample.
    """

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        super().__init__(name="bertscore")
        self.model_type = model_type
        self._scorer = None

    @property
    def expected_keys(self) -> List[str]:
        return ["bertscore_precision", "bertscore_recall", "bertscore_f1"]

    def _get_scorer(self):
        if self._scorer is None:
            _patch_tokenizer_overflow()
            from bert_score import BERTScorer
            self._scorer = BERTScorer(
                model_type=self.model_type,
                device="cuda",
                rescale_with_baseline=False,
            )
            logger.info(f"BERTScorer initialized with model: {self.model_type}")
        return self._scorer

    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        if reference is None:
            raise ValueError("BERTScoreMetric requires a reference (labeled summary).")

        scorer = self._get_scorer()
        P, R, F1 = scorer.score(
            cands=[prediction],
            refs=[reference],
            verbose=False,
        )
        return {
            "bertscore_precision": P[0].item(),
            "bertscore_recall": R[0].item(),
            "bertscore_f1": F1[0].item(),
        }
