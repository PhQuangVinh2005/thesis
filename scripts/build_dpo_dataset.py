#!/usr/bin/env python3
"""Build DPO preference pairs from baseline predictions + faithfulness scores.

For each sample_id across 5 models:
  1. Compute composite faithfulness score per model
  2. Select chosen (highest) and rejected (lowest)
  3. Filter pairs with insufficient score gap

Uses only stdlib — runs in any env.

Usage:
    python scripts/build_dpo_dataset.py

    # Custom weights / threshold:
    python scripts/build_dpo_dataset.py --align-weight 0.5 --summac-weight 0.5 --min-gap 0.10

    # Custom paths:
    python scripts/build_dpo_dataset.py \
        --predictions-dir outputs/baseline/ \
        --faith-dir data/raw/finetune/dpo/baseline/ \
        --output data/processed/dpo/preference_pairs.jsonl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, median


# ── Defaults ──────────────────────────────────────────────────────────

MODELS = ["biomistral7b", "biomistral7b_slerp", "qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b"]
RANGES = ["range_0_1k", "range_1k_2k", "range_2k_4k"]

DEFAULT_PREDICTIONS_DIR = "outputs/baseline"
DEFAULT_FAITH_DIR = "data/raw/finetune/dpo/baseline"
DEFAULT_OUTPUT = "data/processed/dpo/preference_pairs.jsonl"
DEFAULT_ALIGN_WEIGHT = 0.6
DEFAULT_SUMMAC_WEIGHT = 0.4
DEFAULT_MIN_GAP = 0.05
DEFAULT_MAX_LENGTH_RATIO = 0.5  # summary/context ratio; above this = copy-paste


# ── Data loading ──────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_predictions(predictions_dir: Path, model: str, range_id: str) -> dict:
    """Load predictions.jsonl, return dict keyed by sample_id."""
    path = predictions_dir / model / range_id / "predictions.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
        return {}
    by_id = {}
    for rec in load_jsonl(path):
        by_id[rec["sample_id"]] = rec
    return by_id


def load_faith_scores(faith_dir: Path, model: str, range_id: str) -> dict:
    """Load faith_scores.jsonl, return dict keyed by sample_id."""
    path = faith_dir / model / range_id / "faith_scores.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
        return {}
    by_id = {}
    for rec in load_jsonl(path):
        by_id[rec["sample_id"]] = rec
    return by_id


# ── Core logic ────────────────────────────────────────────────────────

def compute_composite(faith_record: dict, align_weight: float, summac_weight: float) -> float | None:
    """Compute composite faithfulness score from a faith_scores record.

    Returns None if required fields are missing.
    """
    alignscore = faith_record.get("alignscore")
    summac_conv = faith_record.get("summac_conv")
    if alignscore is None or summac_conv is None:
        return None
    return align_weight * alignscore + summac_weight * summac_conv


def build_pairs(
    predictions_dir: Path,
    faith_dir: Path,
    align_weight: float,
    summac_weight: float,
    min_gap: float,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
) -> tuple[list[dict], dict]:
    """Build preference pairs from predictions + faith scores.

    Returns:
        pairs: List of preference pair dicts
        stats: Dataset statistics dict
    """
    pairs = []
    skipped_missing = 0
    skipped_gap = 0
    skipped_degenerate = 0  # samples where all non-degenerate models < 2
    degenerate_predictions = 0  # individual model predictions excluded
    total_candidates = 0

    # Track stats
    chosen_model_freq = defaultdict(int)
    rejected_model_freq = defaultdict(int)
    per_range_count = defaultdict(int)
    chosen_scores = []
    rejected_scores = []
    gaps = []
    all_composites = []

    for range_id in RANGES:
        range_short = range_id.replace("range_", "")
        print(f"\n  Processing {range_id}...")

        # Load all models' data for this range
        model_predictions = {}
        model_faith = {}
        for model in MODELS:
            model_predictions[model] = load_predictions(predictions_dir, model, range_id)
            model_faith[model] = load_faith_scores(faith_dir, model, range_id)

        # Get sample_ids that have faith scores in ALL models
        sample_ids_per_model = [set(model_faith[m].keys()) for m in MODELS]
        common_ids = sample_ids_per_model[0]
        for s in sample_ids_per_model[1:]:
            common_ids = common_ids & s

        # Also need predictions for all models
        pred_ids_per_model = [set(model_predictions[m].keys()) for m in MODELS]
        common_pred_ids = pred_ids_per_model[0]
        for s in pred_ids_per_model[1:]:
            common_pred_ids = common_pred_ids & s

        usable_ids = sorted(common_ids & common_pred_ids)
        print(f"    Faith scores common across 5 models: {len(common_ids)}")
        print(f"    Predictions common across 5 models:  {len(common_pred_ids)}")
        print(f"    Usable (both):                       {len(usable_ids)}")

        for sample_id in usable_ids:
            total_candidates += 1

            # Compute composite per model, filtering degenerate outputs
            composites = {}
            context = model_predictions[MODELS[0]][sample_id].get("context", "")
            context_len = len(context)

            for model in MODELS:
                faith_rec = model_faith[model][sample_id]
                score = compute_composite(faith_rec, align_weight, summac_weight)
                if score is None:
                    continue

                # Filter degenerate copy-paste outputs
                pred_summary = model_predictions[model][sample_id].get("predicted_summary", "")
                if context_len > 0 and len(pred_summary) / context_len > max_length_ratio:
                    degenerate_predictions += 1
                    continue

                composites[model] = score
                all_composites.append(score)

            # Need at least 2 non-degenerate models to form a pair
            if len(composites) < 2:
                if any(
                    compute_composite(model_faith[m][sample_id], align_weight, summac_weight) is None
                    for m in MODELS
                ):
                    skipped_missing += 1
                else:
                    skipped_degenerate += 1
                continue

            # Select best (chosen) and worst (rejected)
            best_model = max(composites, key=composites.get)
            worst_model = min(composites, key=composites.get)
            gap = composites[best_model] - composites[worst_model]

            if gap < min_gap:
                skipped_gap += 1
                continue

            # Build the prompt from the prediction record
            # Use any model's prediction to get the context + instruction
            # (all models received the same prompt for the same sample_id)
            ref_pred = model_predictions[best_model][sample_id]
            prompt = ref_pred.get("instruction", "").replace("{context}", ref_pred.get("context", ""))

            # If instruction doesn't contain {context} placeholder (already formatted)
            if ref_pred.get("context", "") and ref_pred["context"] not in prompt:
                prompt = ref_pred.get("instruction", "") + "\n\n" + ref_pred["context"]

            pair = {
                "prompt": prompt,
                "chosen": model_predictions[best_model][sample_id]["predicted_summary"],
                "rejected": model_predictions[worst_model][sample_id]["predicted_summary"],
                # Metadata (ignored by TRL DPOTrainer, useful for analysis)
                "sample_id": sample_id,
                "range": range_short,
                "chosen_model": best_model,
                "rejected_model": worst_model,
                "chosen_score": round(composites[best_model], 6),
                "rejected_score": round(composites[worst_model], 6),
                "score_gap": round(gap, 6),
            }
            pairs.append(pair)

            # Track stats
            chosen_model_freq[best_model] += 1
            rejected_model_freq[worst_model] += 1
            per_range_count[range_short] += 1
            chosen_scores.append(composites[best_model])
            rejected_scores.append(composites[worst_model])
            gaps.append(gap)

    # Build statistics
    stats = {
        "total_samples_across_ranges": total_candidates,
        "total_pairs_after_filter": len(pairs),
        "skipped_missing_scores": skipped_missing,
        "skipped_gap_too_small": skipped_gap,
        "skipped_all_degenerate": skipped_degenerate,
        "degenerate_predictions_excluded": degenerate_predictions,
        "min_gap_threshold": min_gap,
        "max_length_ratio": max_length_ratio,
        "composite_weights": {
            "alignscore": align_weight,
            "summac_conv": summac_weight,
        },
        "per_range": dict(sorted(per_range_count.items())),
        "chosen_model_frequency": dict(sorted(chosen_model_freq.items(), key=lambda x: -x[1])),
        "rejected_model_frequency": dict(sorted(rejected_model_freq.items(), key=lambda x: -x[1])),
        "score_distribution": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if chosen_scores:
        stats["score_distribution"] = {
            "chosen_mean": round(mean(chosen_scores), 6),
            "chosen_std": round(stdev(chosen_scores), 6) if len(chosen_scores) > 1 else 0.0,
            "chosen_median": round(median(chosen_scores), 6),
            "rejected_mean": round(mean(rejected_scores), 6),
            "rejected_std": round(stdev(rejected_scores), 6) if len(rejected_scores) > 1 else 0.0,
            "rejected_median": round(median(rejected_scores), 6),
            "gap_mean": round(mean(gaps), 6),
            "gap_std": round(stdev(gaps), 6) if len(gaps) > 1 else 0.0,
            "gap_median": round(median(gaps), 6),
            "gap_min": round(min(gaps), 6),
            "gap_max": round(max(gaps), 6),
        }

    if all_composites:
        stats["composite_distribution"] = {
            "mean": round(mean(all_composites), 6),
            "std": round(stdev(all_composites), 6) if len(all_composites) > 1 else 0.0,
            "min": round(min(all_composites), 6),
            "max": round(max(all_composites), 6),
        }

    return pairs, stats


# ── Output ────────────────────────────────────────────────────────────

def save_output(pairs: list, stats: dict, output_path: Path) -> None:
    """Save preference pairs (JSONL) and stats (JSON)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save pairs
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\n  Saved pairs: {output_path} ({len(pairs)} records)")

    # Save stats
    stats_path = output_path.parent / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved stats: {stats_path}")


def print_summary(stats: dict) -> None:
    """Print a human-readable summary table."""
    print(f"\n{'='*65}")
    print(f"  DPO Preference Dataset — Summary")
    print(f"{'='*65}")
    print(f"  Total candidate samples:  {stats['total_samples_across_ranges']}")
    print(f"  Pairs after filtering:    {stats['total_pairs_after_filter']}")
    print(f"  Skipped (missing scores): {stats['skipped_missing_scores']}")
    print(f"  Skipped (gap < {stats['min_gap_threshold']}):   {stats['skipped_gap_too_small']}")
    print(f"  Skipped (all degenerate): {stats['skipped_all_degenerate']}")
    print(f"  Degenerate preds excluded:{stats['degenerate_predictions_excluded']}")
    print(f"  Max length ratio:         {stats['max_length_ratio']}")
    print(f"  Composite weights:        {stats['composite_weights']}")

    if stats.get("per_range"):
        print(f"\n  Per range:")
        for r, count in stats["per_range"].items():
            print(f"    {r}: {count} pairs")

    if stats.get("chosen_model_frequency"):
        print(f"\n  Chosen model frequency (who's most faithful):")
        for m, count in stats["chosen_model_frequency"].items():
            pct = 100 * count / stats["total_pairs_after_filter"]
            print(f"    {m:<25} {count:>5} ({pct:5.1f}%)")

    if stats.get("rejected_model_frequency"):
        print(f"\n  Rejected model frequency (who hallucinates most):")
        for m, count in stats["rejected_model_frequency"].items():
            pct = 100 * count / stats["total_pairs_after_filter"]
            print(f"    {m:<25} {count:>5} ({pct:5.1f}%)")

    if stats.get("score_distribution"):
        sd = stats["score_distribution"]
        print(f"\n  Score distribution:")
        print(f"    Chosen:   {sd['chosen_mean']:.4f} ± {sd['chosen_std']:.4f} (median {sd['chosen_median']:.4f})")
        print(f"    Rejected: {sd['rejected_mean']:.4f} ± {sd['rejected_std']:.4f} (median {sd['rejected_median']:.4f})")
        print(f"    Gap:      {sd['gap_mean']:.4f} ± {sd['gap_std']:.4f} (range [{sd['gap_min']:.4f}, {sd['gap_max']:.4f}])")

    print(f"{'='*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs from baseline predictions + faithfulness scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_dpo_dataset.py
  python scripts/build_dpo_dataset.py --min-gap 0.10
  python scripts/build_dpo_dataset.py --align-weight 0.5 --summac-weight 0.5
        """,
    )
    parser.add_argument(
        "--predictions-dir", type=str, default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory with baseline predictions (default: {DEFAULT_PREDICTIONS_DIR})",
    )
    parser.add_argument(
        "--faith-dir", type=str, default=DEFAULT_FAITH_DIR,
        help=f"Directory with faithfulness scores (default: {DEFAULT_FAITH_DIR})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output path for preference pairs JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--align-weight", type=float, default=DEFAULT_ALIGN_WEIGHT,
        help=f"AlignScore weight in composite (default: {DEFAULT_ALIGN_WEIGHT})",
    )
    parser.add_argument(
        "--summac-weight", type=float, default=DEFAULT_SUMMAC_WEIGHT,
        help=f"SummaC_conv weight in composite (default: {DEFAULT_SUMMAC_WEIGHT})",
    )
    parser.add_argument(
        "--min-gap", type=float, default=DEFAULT_MIN_GAP,
        help=f"Minimum composite score gap to keep a pair (default: {DEFAULT_MIN_GAP})",
    )
    parser.add_argument(
        "--max-length-ratio", type=float, default=DEFAULT_MAX_LENGTH_RATIO,
        help=f"Max summary/context length ratio; above this = degenerate copy-paste (default: {DEFAULT_MAX_LENGTH_RATIO})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    predictions_dir = Path(args.predictions_dir)
    faith_dir = Path(args.faith_dir)
    output_path = Path(args.output)

    # Validate inputs
    if not predictions_dir.is_dir():
        print(f"ERROR: Predictions directory not found: {predictions_dir}", file=sys.stderr)
        sys.exit(1)
    if not faith_dir.is_dir():
        print(f"ERROR: Faith scores directory not found: {faith_dir}", file=sys.stderr)
        sys.exit(1)

    # Validate weights sum to ~1.0
    weight_sum = args.align_weight + args.summac_weight
    if abs(weight_sum - 1.0) > 0.01:
        print(f"WARNING: Weights sum to {weight_sum:.2f}, not 1.0", file=sys.stderr)

    print("═══════════════════════════════════════════════════════════════════")
    print("  Building DPO Preference Pairs")
    print("═══════════════════════════════════════════════════════════════════")
    print(f"  Predictions:  {predictions_dir}")
    print(f"  Faith scores: {faith_dir}")
    print(f"  Output:       {output_path}")
    print(f"  Composite:    {args.align_weight} × AlignScore + {args.summac_weight} × SummaC_conv")
    print(f"  Min gap:      {args.min_gap}")
    print(f"  Max len ratio:{args.max_length_ratio}")

    pairs, stats = build_pairs(
        predictions_dir=predictions_dir,
        faith_dir=faith_dir,
        align_weight=args.align_weight,
        summac_weight=args.summac_weight,
        min_gap=args.min_gap,
        max_length_ratio=args.max_length_ratio,
    )

    save_output(pairs, stats, output_path)
    print_summary(stats)


if __name__ == "__main__":
    main()
