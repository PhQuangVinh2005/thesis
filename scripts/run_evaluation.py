#!/usr/bin/env python3
"""
Evaluation runner — completeness (P vs L) and faithfulness (P vs C).

Reads metric params from configs/evaluation.yaml.
Runs AFTER summarization is complete. Does NOT need vLLM server.

Usage:
    # Completeness (default):
    python scripts/run_evaluation.py \
        --predictions outputs/baseline/biomistral7b/range_0_1k/predictions.jsonl

    # Faithfulness:
    python scripts/run_evaluation.py \
        --predictions outputs/baseline/biomistral7b/range_0_1k/predictions.jsonl \
        --phase faithfulness

    # All ranges for a model:
    python scripts/run_evaluation.py \
        --experiment-dir outputs/baseline/biomistral7b/ --phase faithfulness

    # Select specific metrics:
    python scripts/run_evaluation.py \
        --predictions outputs/baseline/biomistral7b/range_0_1k/predictions.jsonl \
        --metrics rouge bleu

    # Debug (5 samples):
    python scripts/run_evaluation.py \
        --predictions outputs/baseline/biomistral7b/range_0_1k/predictions.jsonl \
        --max-samples 5
"""

# ── Output filenames by phase ──────────────────────────────────────────
OUTPUT_FILES = {
    "completeness": ("eval_scores.jsonl", "eval_summary.json"),
    "faithfulness": ("faith_scores.jsonl", "faith_summary.json"),
}
# ───────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import EvalSample
from src.pipelines.evaluator import EvaluationPipeline
from src.utils.io import load_yaml, load_json, load_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Evaluation config path ─────────────────────────────────────────────
EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "evaluation.yaml"


# ── Metric creation ───────────────────────────────────────────────────

def create_metrics(phase: str, metric_names: list, eval_config: dict) -> list:
    """Create metric instances from config. Direct imports, no importlib."""
    instances = []
    phase_config = eval_config.get(phase, {})

    for name in metric_names:
        try:
            metric = _create_single_metric(name, phase_config)
            instances.append(metric)
            logger.info(f"  ✓ {name}")
        except Exception as e:
            logger.warning(f"  ✗ {name}: {e} (skipping)")

    return instances


def _create_single_metric(name: str, phase_config: dict):
    """Create a single metric instance by name."""
    # Completeness metrics (P vs L)
    if name == "rouge":
        from src.evaluation.completeness.rouge import ROUGEMetric
        return ROUGEMetric()
    elif name == "bleu":
        from src.evaluation.completeness.bleu import BLEUMetric
        return BLEUMetric()
    elif name == "bertscore":
        from src.evaluation.completeness.bert_score import BERTScoreMetric
        model_type = phase_config.get("bertscore_model", "microsoft/deberta-xlarge-mnli")
        return BERTScoreMetric(model_type=model_type)
    elif name == "medcon":
        from src.evaluation.completeness.medcon import MEDCONMetric
        model_name = phase_config.get("medcon_model", "en_core_sci_lg")
        return MEDCONMetric(model_name=model_name)
    # Faithfulness metrics (P vs C)
    elif name == "fenice":
        from src.evaluation.faithfulness.fenice import FENICEMetric
        return FENICEMetric()
    elif name == "alignscore":
        from src.evaluation.faithfulness.alignscore import AlignScoreMetric
        as_config = phase_config.get("alignscore", {})
        return AlignScoreMetric(
            ckpt_path=as_config.get("checkpoint", "models/AlignScore-large.ckpt"),
            evaluation_mode=as_config.get("evaluation_mode", "nli_sp"),
            device=as_config.get("device", "auto"),
        )
    elif name == "summac":
        from src.evaluation.faithfulness.summac import SummaCMetric
        sc_config = phase_config.get("summac", {})
        return SummaCMetric(
            granularity=sc_config.get("granularity", "sentence"),
            model_name=sc_config.get("model_name", "vitc"),
            device=sc_config.get("device", "auto"),
        )
    else:
        raise ValueError(f"Unknown metric: '{name}'")


# ── Helpers ───────────────────────────────────────────────────────────

AVAILABLE_METRICS = {
    "completeness": ["rouge", "bleu", "bertscore", "medcon"],
    "faithfulness": ["fenice", "alignscore", "summac"],
}


def predictions_to_samples(records: list) -> list:
    """Convert predictions.jsonl records to EvalSample list."""
    samples = []
    for r in records:
        sample = EvalSample(
            sample_id=r["sample_id"],
            context=r.get("context", ""),
            predicted_summary=r.get("predicted_summary"),
            labeled_summary=r.get("labeled_summary"),
            instruction=r.get("instruction", ""),
            metadata={
                k: v for k, v in r.items()
                if k not in ("sample_id", "context", "predicted_summary",
                             "labeled_summary", "instruction")
            },
        )
        samples.append(sample)
    return samples


def evaluate_single_file(
    predictions_path: str,
    pipeline: EvaluationPipeline,
    max_samples: int | None,
    phase: str = "completeness",
    scores_file_override: str | None = None,
    summary_file_override: str | None = None,
) -> None:
    """Evaluate a single predictions.jsonl file."""
    pred_path = Path(predictions_path)
    output_dir = str(pred_path.parent)

    logger.info(f"Loading predictions: {pred_path}")
    records = load_jsonl(str(pred_path))
    samples = predictions_to_samples(records)
    logger.info(f"Loaded {len(samples)} samples")

    # Read model info from experiment_meta.json
    meta_path = pred_path.parent / "experiment_meta.json"
    model_name = "unknown"
    experiment_name = "experiment"
    prompt_name = "unknown"
    range_id = "unknown"

    if meta_path.exists():
        meta = load_json(str(meta_path))
        model_name = meta.get("model", {}).get("model_name", "unknown")
        experiment_name = meta.get("experiment_name", "experiment")
        prompt_name = meta.get("prompt", {}).get("name", "unknown")
        range_id = meta.get("data", {}).get("range", "unknown")
    else:
        dir_name = pred_path.parent.name
        if dir_name.startswith("range_"):
            range_id = dir_name.replace("range_", "")

    scores_file, summary_file = OUTPUT_FILES[phase]
    if scores_file_override:
        scores_file = scores_file_override
    if summary_file_override:
        summary_file = summary_file_override

    result = pipeline.run(
        samples=samples,
        output_dir=output_dir,
        model_name=model_name,
        range_id=range_id,
        experiment_name=experiment_name,
        prompt_name=prompt_name,
        max_samples=max_samples,
        scores_filename=scores_file,
        summary_filename=summary_file,
    )

    # Print summary table
    phase_label = "Completeness (P vs L)" if phase == "completeness" else "Faithfulness (P vs C)"
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_label}")
    print(f"  Model: {model_name}")
    print(f"  Range: {range_id}")
    print(f"  Samples: {result.num_samples}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Mean':>10}")
    print(f"  {'-'*35}")
    for key, value in sorted(result.scores.items()):
        print(f"  {key:<25} {value:>10.4f}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on predictions")
    parser.add_argument("--phase", type=str, default="completeness",
                        choices=["completeness", "faithfulness"],
                        help="Evaluation phase")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", type=str, help="Single predictions.jsonl file")
    group.add_argument("--experiment-dir", type=str, help="Evaluate all range_*/predictions.jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (debug)")
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics to run (default: all for phase)")
    parser.add_argument("--scores-file", type=str, default=None, help="Override output scores filename")
    parser.add_argument("--summary-file", type=str, default=None, help="Override output summary filename")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load evaluation config
    eval_config = {}
    if EVAL_CONFIG_PATH.exists():
        eval_config = load_yaml(str(EVAL_CONFIG_PATH))
        logger.info(f"Loaded evaluation config: {EVAL_CONFIG_PATH}")

    # Determine metrics to run
    available = AVAILABLE_METRICS[args.phase]
    metric_names = args.metrics if args.metrics else eval_config.get(args.phase, {}).get("metrics", available)

    # Validate
    invalid = [m for m in metric_names if m not in available]
    if invalid:
        logger.error(f"Unknown metrics for phase '{args.phase}': {invalid}. Available: {available}")
        sys.exit(1)

    # Create metric instances
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Loading metrics: {metric_names}")
    metric_instances = create_metrics(args.phase, metric_names, eval_config)

    if not metric_instances:
        logger.error("No metrics could be loaded. Exiting.")
        sys.exit(1)

    pipeline = EvaluationPipeline(metrics=metric_instances)

    if args.predictions:
        evaluate_single_file(
            args.predictions, pipeline, args.max_samples, args.phase,
            scores_file_override=args.scores_file,
            summary_file_override=args.summary_file,
        )
    else:
        exp_dir = Path(args.experiment_dir)
        pred_files = sorted(exp_dir.glob("range_*/predictions.jsonl"))

        if not pred_files:
            logger.error(f"No predictions.jsonl found in {exp_dir}/range_*/")
            sys.exit(1)

        logger.info(f"Found {len(pred_files)} range(s) to evaluate")
        for pred_file in pred_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {pred_file}")
            logger.info(f"{'='*60}")
            evaluate_single_file(
                str(pred_file), pipeline, args.max_samples, args.phase,
                scores_file_override=args.scores_file,
                summary_file_override=args.summary_file,
            )

    logger.info("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
