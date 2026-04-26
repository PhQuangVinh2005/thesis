"""Data schemas for the clinical text summarization pipeline.

Core variables: C (Context), I (Instruction), P (Predicted), L (Labeled).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MedicalRecord:
    """Raw medical record with metadata."""
    record_id: str
    context: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalSample:
    """Single evaluation sample with the 4 core variables (C, I, P, L)."""
    sample_id: str
    context: str                                     # C
    instruction: str = ""                            # I
    predicted_summary: Optional[str] = None          # P
    labeled_summary: Optional[str] = None            # L
    metadata: dict = field(default_factory=dict)

    @property
    def has_prediction(self) -> bool:
        return self.predicted_summary is not None

    @property
    def has_ground_truth(self) -> bool:
        return self.labeled_summary is not None


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_name: str
    model_name: str
    prompt_name: str
    num_samples: int = 0
    scores: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
