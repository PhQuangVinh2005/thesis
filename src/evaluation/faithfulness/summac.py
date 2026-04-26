"""SummaC — NLI-based factual consistency evaluation (P vs C).

Two variants computed per sample:
- SummaC-ZS  (Zero-Shot):  max/mean aggregation of sentence-pair NLI scores
- SummaC-Conv (Convolutional): learned conv layer over NLI score distribution

Based on: Laban et al., TACL 2022.
Package: pip install summac

Performance note:
    ZS and Conv share the same SummaCImager instance so that the NLI
    image matrix is computed once and cached.  ZS must run first in
    compute() to populate the cache before Conv reuses it.
    This halves GPU inference time and VRAM (~14 GB → ~7 GB).
"""

import logging
from typing import Dict, List, Optional

from ..base import BaseMetric

logger = logging.getLogger(__name__)


class SummaCMetric(BaseMetric):
    """SummaC — sentence-level NLI factual consistency.

    Returns both ZS and Conv scores per sample:
      {"summac_zs": float, "summac_conv": float}

    Config params (from evaluation.yaml → faithfulness.summac):
    - granularity: "sentence" (default) or "paragraph"
    - model_name: NLI model tag, default "vitc"
    - device: "auto" | "cuda" | "cpu"
    """

    def __init__(
        self,
        granularity: str = "sentence",
        model_name: str = "vitc",
        device: str = "auto",
    ):
        super().__init__(name="summac")
        self.granularity = granularity
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._zs = None
        self._conv = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def expected_keys(self) -> List[str]:
        return ["summac_zs", "summac_conv"]

    def _load_models(self) -> None:
        """Lazy-load both SummaC scorers on first use.

        Key optimization: Conv reuses ZS's SummaCImager instance so the
        NLI model is loaded once and image cache is shared.  This cuts
        inference time in half and halves VRAM usage.
        """
        if self._zs is not None:
            return

        from summac.model_summac import SummaCZS, SummaCConv

        logger.info(
            f"Loading SummaC models (model={self.model_name}, "
            f"granularity={self.granularity}, device={self.device})"
        )

        # 1. Load ZS first — this creates the NLI model + imager
        self._zs = SummaCZS(
            granularity=self.granularity,
            model_name=self.model_name,
            device=self.device,
        )

        # 2. Load Conv
        self._conv = SummaCConv(
            models=[self.model_name],
            bins="percentile",
            granularity=self.granularity,
            nli_labels="e",
            device=self.device,
            start_file="default",
            agg="mean",
        )

        # 3. Share imager: Conv reuses ZS's NLI model + cache
        #    Both use the same model_name so the imager is identical.
        #    This avoids loading ALBERT-XLarge twice (~7 GB saved)
        #    and lets Conv skip NLI inference via cached images.
        self._conv.imagers[0] = self._zs.imager
        logger.info(
            "SummaC models loaded (ZS + Conv, shared NLI imager)"
        )

    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        """Score a single prediction against its source context.

        Args:
            prediction: Generated summary text (P).
            reference: Labeled summary (L) — unused for faithfulness.
            context: Source clinical notes (C) — required.

        Returns:
            {"summac_zs": float, "summac_conv": float}
        """
        if context is None:
            raise ValueError("SummaCMetric requires context (C).")

        self._load_models()

        # ZS runs FIRST to populate the shared imager cache
        score_zs = self._zs.score([context], [prediction])
        # Conv reuses cached NLI images — no redundant inference
        score_conv = self._conv.score([context], [prediction])

        return {
            "summac_zs": score_zs["scores"][0],
            "summac_conv": score_conv["scores"][0],
        }
