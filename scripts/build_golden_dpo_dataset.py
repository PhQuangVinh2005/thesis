#!/usr/bin/env python3
"""Build DPO preference pairs from expert-annotated golden dataset.

Uses the Hegselmann et al. (CHIL 2024) PhysioNet dataset:
- Original summaries (with hallucinations) → rejected
- Cleaned+Improved summaries (hallucinations removed) → chosen
- Same clinical context (BHC text) → prompt

Creates nested subsets: 10 ⊂ 50 ⊂ 100 for ablation study.

Usage:
    python scripts/build_golden_dpo_dataset.py
    python scripts/build_golden_dpo_dataset.py --seed 42
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev


# ── Defaults ──────────────────────────────────────────────────────────

DATASET_ROOT = Path(
    "data/raw/medical-expert-annotations-of-unsupported-facts-in-doctor-written-and-llm-generated-patient-summaries-1.0.1"
)
DERIVED = DATASET_ROOT / "derived_datasets"

ORIGINAL_PATH = DERIVED / "hallucinations_mimic_di_original.json"
CLEANED_PATH = DERIVED / "hallucinations_mimic_di_cleaned_improved.json"
VALID_ORIGINAL_PATH = DERIVED / "hallucinations_mimic_di_validation_original.json"
VALID_CLEANED_PATH = DERIVED / "hallucinations_mimic_di_validation_cleaned_improved.json"

OUTPUT_DIR = Path("data/processed/dpo/golden")

INSTRUCTION = (
    "You are an expert medical professional specializing in clinical documentation.\n\n"
    "Summarize the following clinical notes into a concise Brief Hospital Course. "
    "Focus only on clinically significant events, diagnoses, and treatments. "
    "Ensure the summary is technically accurate and uses professional medical terminology.\n\n"
    "Strictly adhere to the provided context. If a piece of information is not explicitly "
    "stated in the input text, do not include it. Avoid inferring results or patient outcomes.\n\n"
    "input: {context}\nsummary:\n"
)

SUBSET_SIZES = [10, 50, 100]
DEFAULT_SEED = 42


# ── Data loading ──────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_pairs_from_files(
    original_path: Path,
    cleaned_path: Path,
    instruction: str,
) -> list[dict]:
    """Build preference pairs from original (hallucinated) and cleaned (faithful) summaries.

    Returns list of {prompt, chosen, rejected, metadata...} dicts.
    """
    originals = load_jsonl(original_path)
    cleaned = load_jsonl(cleaned_path)

    if len(originals) != len(cleaned):
        print(
            f"ERROR: Mismatched record counts: original={len(originals)}, cleaned={len(cleaned)}",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs = []
    for i, (orig, clean) in enumerate(zip(originals, cleaned)):
        # Verify same context
        if orig["text"] != clean["text"]:
            print(f"WARNING: Context mismatch at index {i}, skipping", file=sys.stderr)
            continue

        context = orig["text"]
        prompt = instruction.replace("{context}", context)

        pair = {
            "prompt": prompt,
            "chosen": clean["summary"],
            "rejected": orig["summary"],
            # Metadata
            "index": i,
            "context_len": len(context),
            "chosen_len": len(clean["summary"]),
            "rejected_len": len(orig["summary"]),
            "len_diff": len(orig["summary"]) - len(clean["summary"]),
        }
        pairs.append(pair)

    return pairs


# ── Output ────────────────────────────────────────────────────────────

def compute_stats(pairs: list[dict], label: str) -> dict:
    """Compute summary statistics for a set of pairs."""
    if not pairs:
        return {"label": label, "count": 0}

    context_lens = [p["context_len"] for p in pairs]
    chosen_lens = [p["chosen_len"] for p in pairs]
    rejected_lens = [p["rejected_len"] for p in pairs]
    len_diffs = [p["len_diff"] for p in pairs]

    return {
        "label": label,
        "count": len(pairs),
        "context_len": {
            "mean": round(mean(context_lens), 1),
            "std": round(stdev(context_lens), 1) if len(context_lens) > 1 else 0,
            "min": min(context_lens),
            "max": max(context_lens),
        },
        "chosen_len": {
            "mean": round(mean(chosen_lens), 1),
            "std": round(stdev(chosen_lens), 1) if len(chosen_lens) > 1 else 0,
        },
        "rejected_len": {
            "mean": round(mean(rejected_lens), 1),
            "std": round(stdev(rejected_lens), 1) if len(rejected_lens) > 1 else 0,
        },
        "len_diff_rejected_minus_chosen": {
            "mean": round(mean(len_diffs), 1),
            "positive_count": sum(1 for d in len_diffs if d > 0),
            "description": "Positive = rejected is longer (hallucinations add text)",
        },
    }


def save_pairs(pairs: list[dict], path: Path) -> None:
    """Save pairs as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  Saved: {path} ({len(pairs)} pairs)")


def main():
    parser = argparse.ArgumentParser(
        description="Build golden DPO preference pairs from expert-annotated hallucination dataset",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for shuffling (default: {DEFAULT_SEED})")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("═══════════════════════════════════════════════════════════════════")
    print("  Building Golden DPO Preference Pairs")
    print("  Source: Hegselmann et al. (CHIL 2024) PhysioNet dataset")
    print("═══════════════════════════════════════════════════════════════════")

    # ── Build training pairs ──
    print("\n  Loading training pairs...")
    all_pairs = build_pairs_from_files(ORIGINAL_PATH, CLEANED_PATH, INSTRUCTION)
    print(f"    Total: {len(all_pairs)} pairs")

    # Shuffle with fixed seed for reproducibility
    random.seed(args.seed)
    random.shuffle(all_pairs)

    # Re-index after shuffle
    for i, pair in enumerate(all_pairs):
        pair["index"] = i

    # Create nested subsets
    subsets = {}
    for size in SUBSET_SIZES:
        subset = all_pairs[:size]
        subsets[size] = subset
        save_pairs(subset, output_dir / f"preference_pairs_{size}.jsonl")

    # ── Build validation pairs ──
    print("\n  Loading validation pairs...")
    if VALID_ORIGINAL_PATH.exists() and VALID_CLEANED_PATH.exists():
        val_pairs = build_pairs_from_files(VALID_ORIGINAL_PATH, VALID_CLEANED_PATH, INSTRUCTION)
        save_pairs(val_pairs, output_dir / "validation_pairs.jsonl")
        print(f"    Total: {len(val_pairs)} validation pairs")
    else:
        val_pairs = []
        print("    WARNING: Validation files not found, skipping")

    # ── Compute stats ──
    all_stats = {
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(DATASET_ROOT),
        "instruction_template": INSTRUCTION[:80] + "...",
        "subsets": {},
    }

    for size in SUBSET_SIZES:
        all_stats["subsets"][f"train_{size}"] = compute_stats(subsets[size], f"train_{size}")

    if val_pairs:
        all_stats["validation"] = compute_stats(val_pairs, "validation")

    # Verify nesting
    ids_10 = [p["index"] for p in subsets[10]]
    ids_50 = [p["index"] for p in subsets[50]]
    ids_100 = [p["index"] for p in subsets[100]]
    all_stats["nesting_verified"] = {
        "10_in_50": all(i in ids_50 for i in ids_10),
        "50_in_100": all(i in ids_100 for i in ids_50),
    }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved stats: {stats_path}")

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  Golden DPO Dataset — Summary")
    print(f"{'='*65}")
    for size in SUBSET_SIZES:
        s = all_stats["subsets"][f"train_{size}"]
        print(f"  train_{size:>3}: {s['count']:>3} pairs | ctx {s['context_len']['mean']:.0f}c | "
              f"chosen {s['chosen_len']['mean']:.0f}c | rejected {s['rejected_len']['mean']:.0f}c")
    if val_pairs:
        v = all_stats["validation"]
        print(f"  valid:    {v['count']:>3} pairs | ctx {v['context_len']['mean']:.0f}c | "
              f"chosen {v['chosen_len']['mean']:.0f}c | rejected {v['rejected_len']['mean']:.0f}c")
    n = all_stats["nesting_verified"]
    print(f"\n  Nesting: 10⊂50={n['10_in_50']}, 50⊂100={n['50_in_100']}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
