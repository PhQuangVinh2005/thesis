#!/usr/bin/env python3
"""Merge per-metric faithfulness score files into unified output.

Joins summac_scores.jsonl + align_scores.jsonl on sample_id →
  faith_scores.jsonl (per-sample) + faith_summary.json (aggregate)

Uses only stdlib — runs in any env (no deps required).

Usage:
    python scripts/merge_faith_scores.py outputs/baseline/qwen3_5_2b/range_0_1k/
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev
import time


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def merge_scores(output_dir: Path) -> None:
    summac_path = output_dir / "summac_scores.jsonl"
    align_path = output_dir / "align_scores.jsonl"

    if not summac_path.exists() and not align_path.exists():
        print(f"ERROR: No score files found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Load available scores
    summac_by_id = {}
    align_by_id = {}

    if summac_path.exists():
        for r in load_jsonl(summac_path):
            summac_by_id[r["sample_id"]] = r
        print(f"  Loaded {len(summac_by_id)} SummaC scores")

    if align_path.exists():
        for r in load_jsonl(align_path):
            align_by_id[r["sample_id"]] = r
        print(f"  Loaded {len(align_by_id)} AlignScore scores")

    # Merge on sample_id
    all_ids = sorted(set(list(summac_by_id.keys()) + list(align_by_id.keys())))
    merged = []

    for sid in all_ids:
        record = {"sample_id": sid}
        if sid in summac_by_id:
            for k, v in summac_by_id[sid].items():
                if k != "sample_id":
                    record[k] = v
        if sid in align_by_id:
            for k, v in align_by_id[sid].items():
                if k != "sample_id":
                    record[k] = v
        merged.append(record)

    # Save merged per-sample scores
    faith_scores_path = output_dir / "faith_scores.jsonl"
    with open(faith_scores_path, "w") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved: {faith_scores_path} ({len(merged)} samples)")

    # Compute aggregate stats
    metric_keys = set()
    for r in merged:
        metric_keys.update(k for k in r if k != "sample_id")

    aggregate = {}
    for key in sorted(metric_keys):
        values = [r[key] for r in merged if r.get(key) is not None]
        if values:
            aggregate[key] = {
                "mean": round(mean(values), 6),
                "std": round(stdev(values), 6) if len(values) > 1 else 0.0,
            }

    # Save aggregate summary
    summary = {
        "phase": "faithfulness",
        "num_samples": len(merged),
        "metrics_used": sorted(metric_keys),
        "metrics": aggregate,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    faith_summary_path = output_dir / "faith_summary.json"
    with open(faith_summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {faith_summary_path}")

    # Print table
    print(f"\n{'='*60}")
    print(f"  Faithfulness (P vs C) — Merged Results")
    print(f"  Samples: {len(merged)}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*45}")
    for key, stats in sorted(aggregate.items()):
        print(f"  {key:<25} {stats['mean']:>10.4f} {stats['std']:>10.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Merge faithfulness score files")
    parser.add_argument("output_dir", help="Directory containing summac_scores.jsonl and/or align_scores.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"ERROR: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Merging faithfulness scores in: {output_dir}")
    merge_scores(output_dir)


if __name__ == "__main__":
    main()
