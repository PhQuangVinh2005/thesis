"""Evaluation metrics — lazy imports to avoid heavy deps at import time."""

from .base import BaseMetric

__all__ = [
    "BaseMetric",
    "ROUGEMetric", "BLEUMetric", "BERTScoreMetric", "MEDCONMetric",
    "FENICEMetric", "AlignScoreMetric", "SummaCMetric",
]


def __getattr__(name: str):
    """Lazy import metric classes."""
    _METRIC_MAP = {
        "ROUGEMetric": ".completeness.rouge",
        "BLEUMetric": ".completeness.bleu",
        "BERTScoreMetric": ".completeness.bert_score",
        "MEDCONMetric": ".completeness.medcon",
        "FENICEMetric": ".faithfulness.fenice",
        "AlignScoreMetric": ".faithfulness.alignscore",
        "SummaCMetric": ".faithfulness.summac",
    }
    if name in _METRIC_MAP:
        import importlib
        module = importlib.import_module(_METRIC_MAP[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
