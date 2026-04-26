"""MEDCON metric — medical concept F1 overlap using SciSpaCy (P vs L)."""

import logging
from typing import Dict, List, Optional, Set

from ..base import BaseMetric

logger = logging.getLogger(__name__)


class MEDCONMetric(BaseMetric):
    """Medical Concept F1 overlap via en_core_sci_lg NER.

    Extracts medical entities, normalizes, computes set-based P/R/F1.
    Model name configurable via evaluation.yaml.
    """

    def __init__(self, model_name: str = "en_core_sci_lg"):
        super().__init__(name="medcon")
        self.model_name = model_name
        self._nlp = None

    @property
    def expected_keys(self) -> List[str]:
        return ["medcon_precision", "medcon_recall", "medcon_f1"]

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.model_name)
            logger.info(f"MEDCON: loaded SciSpaCy model '{self.model_name}'")
        return self._nlp

    def _extract_concepts(self, text: str) -> Set[str]:
        nlp = self._get_nlp()
        doc = nlp(text)
        concepts = set()
        for ent in doc.ents:
            normalized = ent.text.lower().strip()
            if len(normalized) >= 2 and normalized != "___":
                concepts.add(normalized)
        return concepts

    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        if reference is None:
            raise ValueError("MEDCONMetric requires a reference (labeled summary).")

        pred_concepts = self._extract_concepts(prediction)
        ref_concepts = self._extract_concepts(reference)

        if not pred_concepts and not ref_concepts:
            return {"medcon_precision": 1.0, "medcon_recall": 1.0, "medcon_f1": 1.0}
        if not pred_concepts or not ref_concepts:
            return {"medcon_precision": 0.0, "medcon_recall": 0.0, "medcon_f1": 0.0}

        overlap = pred_concepts & ref_concepts
        precision = len(overlap) / len(pred_concepts)
        recall = len(overlap) / len(ref_concepts)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "medcon_precision": precision,
            "medcon_recall": recall,
            "medcon_f1": f1,
        }
