"""
Data loader for MIMIC-IV-BHC summarization dataset.

Loads preprocessed JSONL files (from scripts/preprocess_mimic_iv_bhc.py)
and converts them to EvalSample objects for the summarization pipeline.
"""

import re
from pathlib import Path
from typing import List, Optional

from .base import BaseDataLoader
from .schema import EvalSample

# Valid range identifiers matching preprocessed JSONL filenames
VALID_RANGES = ("0_1k", "1k_2k", "2k_4k")


class MIMICBHCLoader(BaseDataLoader):
    """
    Loader for MIMIC-IV-BHC preprocessed JSONL data.

    Expected JSONL fields:
        - note_id: str         → sample_id
        - input: str           → context (C)
        - target: str          → labeled_summary (L)
        - input_tokens_gpt4: int
        - target_tokens_gpt4: int

    Usage:
        loader = MIMICBHCLoader("data/processed/mimic_iv_bhc")
        samples = loader.load(range_id="0_1k")
        sampled = loader.sample(n=384, seed=42)
    """

    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, **kwargs)

    def load(
        self,
        range_id: str = "0_1k",
        max_samples: Optional[int] = None,
    ) -> List[EvalSample]:
        """
        Load JSONL file for a specific token range.

        Args:
            range_id: One of "0_1k", "1k_2k", "2k_4k".
            max_samples: Limit records loaded (for debugging). None = all.

        Returns:
            List of EvalSamples.
        """
        if range_id not in VALID_RANGES:
            raise ValueError(
                f"Invalid range_id '{range_id}'. Must be one of: {VALID_RANGES}"
            )

        filepath = Path(self.data_dir) / f"range_{range_id}.jsonl"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Lazy import to avoid circular deps
        from ..utils.io import load_jsonl

        raw_records = load_jsonl(str(filepath))

        if max_samples is not None:
            raw_records = raw_records[:max_samples]

        self._samples = [self._to_eval_sample(r) for r in raw_records]
        return self._samples

    def _to_eval_sample(self, record: dict) -> EvalSample:
        """Convert a raw JSONL record to an EvalSample."""
        return EvalSample(
            sample_id=record["note_id"],
            context=self.preprocess(record["input"]),
            labeled_summary=record.get("target"),
            metadata={
                "input_tokens_gpt4": record.get("input_tokens_gpt4"),
                "target_tokens_gpt4": record.get("target_tokens_gpt4"),
            },
        )

    def preprocess(self, raw_text: str) -> str:
        """
        Normalize clinical text while preserving key features.

        Steps:
        1. Normalize whitespace (collapse multiple spaces/newlines)
        2. Strip leading/trailing whitespace
        3. Keep de-id tokens '___' intact (hallucination traps)
        4. Keep section tags <SECTION_NAME> intact
        """
        # Collapse multiple whitespace to single space
        text = re.sub(r"[ \t]+", " ", raw_text)
        # Normalize multiple newlines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


