#!/usr/bin/env python3
"""Add severity scores to golden DPO preference pairs.

Reads hallucinations_mimic_di.jsonl (expert annotations) and attaches
per-sample severity scores to preference_pairs_100.jsonl, producing
preference_pairs_100_severity.jsonl for SW-DPO training.

Severity weights are derived from the NCC MERP patient safety index:
  contradicted_fact=5.0, medication_unsupported=4.0, condition_unsupported=3.5,
  procedure_unsupported=3.0, number_unsupported=3.0, time_unsupported=2.0,
  location_unsupported=2.0, name_unsupported=1.5, word_unsupported=1.0,
  other_unsupported=1.0

Usage:
    python scripts/add_severity_to_dpo_dataset.py
    python scripts/add_severity_to_dpo_dataset.py --alpha 0.5
    python scripts/add_severity_to_dpo_dataset.py --dry-run

Output:
    data/processed/dpo/golden/preference_pairs_100_severity.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev


# ── Paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_PATH = PROJECT_ROOT / (
    "data/raw/medical-expert-annotations-of-unsupported-facts-in-doctor-written-and-"
    "llm-generated-patient-summaries-1.0.1/hallucination_datasets/hallucinations_mimic_di.jsonl"
)

GOLDEN_INPUT_PATH = PROJECT_ROOT / "data/processed/dpo/golden/preference_pairs_100.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/dpo/golden/preference_pairs_100_severity.jsonl"


# ── Severity weights (NCC MERP patient safety framework) ──────────────

SEVERITY_WEIGHTS: dict[str, float] = {
    "contradicted_fact":     5.0,  # Direct contradiction → wrong treatment risk
    "medication_unsupported": 4.0,  # Wrong drug/dose → adverse events
    "condition_unsupported":  3.5,  # Wrong diagnosis → wrong treatment plan
    "procedure_unsupported":  3.0,  # Wrong procedure → misleading follow-up
    "number_unsupported":     3.0,  # Wrong lab value → clinical decisions
    "time_unsupported":       2.0,  # Wrong timing → scheduling errors
    "location_unsupported":   2.0,  # Wrong body part → usually obvious
    "name_unsupported":       1.5,  # Wrong specialist → low clinical impact
    "word_unsupported":       1.0,  # Stylistic/word-level → minimal risk
    "other_unsupported":      1.0,  # Catch-all → minimal impact
}


# ── Helpers ───────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_context_from_prompt(prompt: str) -> str | None:
    """Extract the raw clinical context text from a formatted DPO prompt.

    The prompt format is:
        ...instruction...\n\ninput: {context}\nsummary:\n
    """
    marker = "input: "
    end_marker = "\nsummary:\n"
    start_idx = prompt.find(marker)
    if start_idx == -1:
        return None
    start_idx += len(marker)
    end_idx = prompt.find(end_marker, start_idx)
    if end_idx == -1:
        return None
    return prompt[start_idx:end_idx]


def compute_severity(labels: list[dict]) -> tuple[float, dict[str, int]]:
    """Compute raw severity score and per-category span counts for one sample.

    Args:
        labels: list of annotation dicts with 'label' key.

    Returns:
        (severity_score, category_counts)
    """
    counts: Counter = Counter()
    for span in labels:
        cat = span.get("label", "other_unsupported")
        counts[cat] += 1

    severity_score = sum(
        SEVERITY_WEIGHTS.get(cat, 1.0) * cnt
        for cat, cnt in counts.items()
    )
    return severity_score, dict(counts)


# ── Main ──────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Load data ──────────────────────────────────────────────────────
    print("Loading annotation data...")
    if not ANNOTATION_PATH.exists():
        print(f"ERROR: Annotation file not found: {ANNOTATION_PATH}", file=sys.stderr)
        sys.exit(1)
    hal_records = load_jsonl(ANNOTATION_PATH)
    print(f"  Loaded {len(hal_records)} annotated records from hallucinations_mimic_di.jsonl")

    print("Loading golden DPO pairs...")
    if not GOLDEN_INPUT_PATH.exists():
        print(f"ERROR: Golden pairs not found: {GOLDEN_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    golden_pairs = load_jsonl(GOLDEN_INPUT_PATH)
    print(f"  Loaded {len(golden_pairs)} golden preference pairs")

    # ── Build lookup: text → labels ────────────────────────────────────
    print("\nBuilding text → labels lookup...")
    text_to_labels: dict[str, list[dict]] = {}
    for rec in hal_records:
        text_to_labels[rec["text"]] = rec["labels"]
    print(f"  Lookup size: {len(text_to_labels)} unique clinical texts")

    # ── Match and compute severity ─────────────────────────────────────
    print("\nMatching golden pairs to annotations and computing severity...")
    enriched_pairs = []
    n_matched = 0
    n_unmatched = 0

    for pair in golden_pairs:
        context = extract_context_from_prompt(pair["prompt"])
        if context is None:
            print(f"  WARNING: Could not extract context from prompt at index {pair['index']}")
            n_unmatched += 1
            severity_score = 0.0
            category_counts: dict[str, int] = {}
        elif context in text_to_labels:
            labels = text_to_labels[context]
            severity_score, category_counts = compute_severity(labels)
            n_matched += 1
        else:
            print(f"  WARNING: No annotation match for pair index {pair['index']}")
            n_unmatched += 1
            severity_score = 0.0
            category_counts = {}

        enriched_pairs.append({
            **pair,
            "severity_score": severity_score,
            "category_counts": category_counts,
            # norm_severity and severity_margin filled after we know max
        })

    print(f"  Matched: {n_matched} / {len(golden_pairs)}")
    if n_unmatched > 0:
        print(f"  WARNING: {n_unmatched} pairs had no match → severity_score=0.0")

    # ── Normalize severity scores ──────────────────────────────────────
    all_scores = [p["severity_score"] for p in enriched_pairs]
    max_score = max(all_scores) if all_scores else 1.0
    min_score = min(all_scores)
    mean_score = mean(all_scores)
    std_score = stdev(all_scores) if len(all_scores) > 1 else 0.0

    print("\n=== Severity Score Distribution ===")
    print(f"  min={min_score:.2f}  max={max_score:.2f}  "
          f"mean={mean_score:.2f}  std={std_score:.2f}")
    print(f"  α={args.alpha}  → severity_margin = α × norm_severity")

    # Add normalized fields
    for pair in enriched_pairs:
        norm = pair["severity_score"] / max_score if max_score > 0 else 0.0
        pair["norm_severity"] = round(norm, 6)
        pair["severity_margin"] = round(args.alpha * norm, 6)
        pair["severity_max_in_dataset"] = max_score
        pair["severity_alpha"] = args.alpha

    # ── Print histogram ────────────────────────────────────────────────
    print("\n=== Severity Distribution (normalized) ===")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in bins:
        count = sum(1 for p in enriched_pairs if lo <= p["norm_severity"] < hi)
        bar = "█" * count
        print(f"  [{lo:.1f}-{hi:.1f}): {bar} {count}")
    # Exactly 1.0
    count_max = sum(1 for p in enriched_pairs if p["norm_severity"] == 1.0)
    if count_max:
        print(f"  [1.0    ]: {'█' * count_max} {count_max}")

    print("\n=== Category Breakdown (across all matched pairs) ===")
    total_counts: Counter = Counter()
    for p in enriched_pairs:
        total_counts.update(p["category_counts"])
    for cat, cnt in sorted(total_counts.items(), key=lambda x: -x[1]):
        weight = SEVERITY_WEIGHTS.get(cat, 1.0)
        print(f"  {cat:30s}: {cnt:3d} spans  weight={weight}")

    # ── Top/bottom severity examples ──────────────────────────────────
    sorted_by_sev = sorted(enriched_pairs, key=lambda p: -p["severity_score"])
    print("\n=== Top 5 Highest Severity Pairs ===")
    for p in sorted_by_sev[:5]:
        print(f"  index={p['index']:3d}  score={p['severity_score']:.2f}  "
              f"norm={p['norm_severity']:.3f}  margin={p['severity_margin']:.3f}  "
              f"cats={p['category_counts']}")
    print("\n=== 5 Zero-Severity Pairs (no annotation match or clean) ===")
    zeros = [p for p in enriched_pairs if p["severity_score"] == 0.0]
    for p in zeros[:5]:
        print(f"  index={p['index']:3d}  score=0.00  (no annotation)")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(enriched_pairs)} pairs to: {OUTPUT_PATH}")
        print("[DRY RUN] No files written.")
        return

    # ── Save output ────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in enriched_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\n✓ Saved {len(enriched_pairs)} pairs to: {OUTPUT_PATH}")

    # ── Verify output ──────────────────────────────────────────────────
    verify = load_jsonl(OUTPUT_PATH)
    if len(verify) != len(enriched_pairs):
        print(
            f"  ERROR: Output count mismatch: wrote {len(enriched_pairs)}, "
            f"read back {len(verify)}",
            file=sys.stderr,
        )
        sys.exit(1)
    required_fields = {"prompt", "chosen", "rejected", "severity_score",
                       "norm_severity", "category_counts", "severity_margin"}
    missing = required_fields - set(verify[0].keys())
    if missing:
        print(f"  ERROR: Missing fields in output: {missing}", file=sys.stderr)
        sys.exit(1)
    print(f"✓ Verified: {len(verify)} records, all required fields present")
    print(f"✓ Fields: {sorted(verify[0].keys())}")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add NCC MERP severity scores to golden DPO preference pairs"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Severity margin strength α (default: 1.0). severity_margin = α × norm_severity"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print severity stats without writing output file"
    )
    parser.add_argument(
        "--annotation-path", type=str, default=None,
        help="Override annotation JSONL path"
    )
    parser.add_argument(
        "--input-path", type=str, default=None,
        help="Override golden pairs input JSONL path"
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Override output JSONL path"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Apply path overrides
    if args.annotation_path:
        ANNOTATION_PATH = Path(args.annotation_path)
    if args.input_path:
        GOLDEN_INPUT_PATH = Path(args.input_path)
    if args.output_path:
        OUTPUT_PATH = Path(args.output_path)
    main(args)
