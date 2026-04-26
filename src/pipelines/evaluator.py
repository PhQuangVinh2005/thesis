"""
Evaluation pipeline.

Runs configured metrics on generated summaries, outputs:
- Per-sample scores (JSONL)
- Aggregate summary (JSON) with mean±std

Runs AFTER summarization is complete — not in parallel with vLLM server.
"""

import json
import logging
import time
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Optional

from tqdm import tqdm

from ..data.schema import EvalSample, ExperimentResult
from ..evaluation.base import BaseMetric

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Evaluates generated summaries using multiple metrics."""

    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

    def evaluate_single(self, sample: EvalSample) -> Dict[str, float]:
        """Evaluate a single sample across all metrics."""
        scores = {}
        for metric in self.metrics:
            try:
                result = metric.compute(
                    prediction=sample.predicted_summary,
                    reference=sample.labeled_summary,
                    context=sample.context,
                )
                scores.update(result)
            except Exception as e:
                logger.error(
                    f"Metric '{metric.name}' failed on {sample.sample_id}: {e}"
                )
                # Use metric's own expected_keys for error fallback
                for key in metric.expected_keys:
                    scores[key] = None
        return scores

    def run(
        self,
        samples: List[EvalSample],
        output_dir: str,
        model_name: str = "unknown",
        range_id: str = "unknown",
        experiment_name: str = "experiment",
        prompt_name: str = "unknown",
        max_samples: Optional[int] = None,
        scores_filename: str = "eval_scores.jsonl",
        summary_filename: str = "eval_summary.json",
    ) -> ExperimentResult:
        """Run evaluation on all samples and save results."""
        if max_samples is not None:
            samples = samples[:max_samples]

        # Filter to samples with both prediction and reference
        valid = [
            s for s in samples
            if s.predicted_summary and s.labeled_summary
            and not s.predicted_summary.startswith("[ERROR]")
        ]
        logger.info(
            f"Evaluating {len(valid)} valid samples "
            f"(skipped {len(samples) - len(valid)} without prediction/reference)"
        )

        start_time = time.time()
        all_scores: List[Dict] = []

        for sample in tqdm(valid, desc="Evaluating"):
            scores = self.evaluate_single(sample)
            all_scores.append({"sample_id": sample.sample_id, **scores})

        elapsed = time.time() - start_time

        # Aggregate: mean ± std per metric key
        metric_keys = set()
        for s in all_scores:
            metric_keys.update(k for k in s.keys() if k != "sample_id")

        aggregate = {}
        for key in sorted(metric_keys):
            values = [s[key] for s in all_scores if s.get(key) is not None]
            if values:
                aggregate[key] = {
                    "mean": round(mean(values), 6),
                    "std": round(stdev(values), 6) if len(values) > 1 else 0.0,
                }

        # Save per-sample scores
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        scores_path = out / scores_filename
        with open(scores_path, "w") as f:
            for record in all_scores:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved per-sample scores: {scores_path}")

        # Save aggregate summary
        summary = {
            "experiment_name": experiment_name,
            "model": model_name,
            "range": range_id,
            "num_samples": len(valid),
            "time_taken_seconds": round(elapsed, 2),
            "metrics_used": [m.name for m in self.metrics],
            "metrics": aggregate,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        summary_path = out / summary_filename
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved eval summary: {summary_path}")

        result = ExperimentResult(
            experiment_name=experiment_name,
            model_name=model_name,
            prompt_name=prompt_name,
            num_samples=len(valid),
            scores={k: v["mean"] for k, v in aggregate.items()},
            samples=all_scores,
        )
        return result
