"""AlignScore — factual consistency via unified alignment function (P vs C).

Pipeline: context chunking (350 tokens) → sentence splitting → NLI alignment.
Based on: Zha et al., 2023 (ACL).
"""

import logging
from typing import Dict, List, Optional

from ..base import BaseMetric

logger = logging.getLogger(__name__)


class AlignScoreMetric(BaseMetric):
    """AlignScore — chunk-based factual consistency evaluation.

    Config params come from evaluation.yaml:
    - checkpoint: path to AlignScore-large.ckpt
    - evaluation_mode: "nli_sp" (default)
    - device: "auto" | "cuda" | "cpu"
    """

    def __init__(
        self,
        ckpt_path: str = "models/AlignScore-large.ckpt",
        evaluation_mode: str = "nli_sp",
        device: str = "auto",
    ):
        super().__init__(name="alignscore")
        self.ckpt_path = ckpt_path
        self.evaluation_mode = evaluation_mode
        self.device = self._resolve_device(device)
        self._scorer = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def expected_keys(self) -> List[str]:
        return ["alignscore"]

    def _get_scorer(self):
        if self._scorer is None:
            from alignscore import AlignScore
            self._scorer = AlignScore(
                model="roberta-large",
                batch_size=8,
                device=self.device,
                ckpt_path=self.ckpt_path,
                evaluation_mode=self.evaluation_mode,
            )
            logger.info(f"AlignScore loaded (device={self.device}, mode={self.evaluation_mode})")
        return self._scorer

    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        if context is None:
            raise ValueError("AlignScoreMetric requires context (C).")

        scorer = self._get_scorer()
        score = scorer.score(contexts=[context], claims=[prediction])
        return {"alignscore": score[0]}
