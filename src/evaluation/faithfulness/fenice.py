"""FENICE — Factuality Evaluation based on NLI and Claim Extraction (P vs C).

Pipeline: claim extraction → coreference resolution → NLI entailment check.
Based on: Scirè et al., 2024.
"""

import logging
from typing import Dict, List, Optional

from ..base import BaseMetric

logger = logging.getLogger(__name__)


class FENICEMetric(BaseMetric):
    """FENICE — claim-level faithfulness with coreference resolution.

    Especially useful for clinical text where pronoun/entity confusion
    is a critical source of hallucination.
    """

    def __init__(self):
        super().__init__(name="fenice")
        self._scorer = None

    @property
    def expected_keys(self) -> List[str]:
        return ["fenice_score"]

    def _get_scorer(self):
        if self._scorer is None:
            from fenice import FENICE
            self._scorer = FENICE()
            logger.info("FENICE scorer loaded")
        return self._scorer

    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        if context is None:
            raise ValueError("FENICEMetric requires context (C).")

        scorer = self._get_scorer()
        result = scorer.score(predictions=[prediction], sources=[context])
        return {"fenice_score": result[0]}
