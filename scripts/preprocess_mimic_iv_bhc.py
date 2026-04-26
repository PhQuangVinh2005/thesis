#!/usr/bin/env python3
"""
Preprocess MIMIC-IV-BHC dataset.

Steps:
  1. Split into 3 context ranges (0-1K, 1K-2K, 2K-4K) by input_tokens
  2. Filter target outliers (50 <= target_tokens <= 2000)
  3. Keep all input sections intact (no pruning)
  4. Keep de-id tokens '___' as-is (hallucination traps)
  5. Sample N records per range (stratified random sampling, seed=42)

Output: JSONL files per range + preprocessing report.

Usage:
    python scripts/preprocess_mimic_iv_bhc.py
    python scripts/preprocess_mimic_iv_bhc.py --sample-size 500
    python scripts/preprocess_mimic_iv_bhc.py --sample-size 0   # no sampling, keep all
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = (
    RAW_DIR
    / "mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0"
    / "mimic-iv-bhc.csv"
)
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "mimic_iv_bhc"

# ── Range definitions (matching paper) ─────────────────────────────────
RANGES = {
    "range_0_1k": (0, 1024),
    "range_1k_2k": (1024, 2048),
    "range_2k_4k": (2048, 4096),
}

# ── Target outlier thresholds ──────────────────────────────────────────
TARGET_MIN = 50
TARGET_MAX = 2000

# ── Sampling ───────────────────────────────────────────────────────────
DEFAULT_SAMPLE_SIZE = 500
SAMPLE_SEED = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV."""
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} records, columns: {list(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    return df


def step1_split_ranges(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split into 3 context-length ranges."""
    splits: dict[str, pd.DataFrame] = {}
    for name, (lo, hi) in RANGES.items():
        mask = (df["input_tokens"] > lo) & (df["input_tokens"] <= hi)
        splits[name] = df[mask].copy()
        print(f"  {name}: {len(splits[name]):,} records")
    return splits


def step2_filter_target(
    splits: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Remove records with target_tokens outside [TARGET_MIN, TARGET_MAX]."""
    filtered: dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        before = len(df)
        mask = (df["target_tokens"] >= TARGET_MIN) & (
            df["target_tokens"] <= TARGET_MAX
        )
        filtered[name] = df[mask].copy()
        after = len(filtered[name])
        removed = before - after
        print(f"  {name}: {before:,} → {after:,}  (removed {removed:,})")
    return filtered


def validate(splits: dict[str, pd.DataFrame]) -> None:
    """Run automated checks."""
    all_ids: list[str] = []
    for name, df in splits.items():
        lo, hi = RANGES[name]
        # Range check
        assert (df["input_tokens"] > lo).all(), f"{name}: input_tokens <= {lo}"
        assert (df["input_tokens"] <= hi).all(), f"{name}: input_tokens > {hi}"
        # Target check
        assert (df["target_tokens"] >= TARGET_MIN).all(), f"{name}: target < {TARGET_MIN}"
        assert (df["target_tokens"] <= TARGET_MAX).all(), f"{name}: target > {TARGET_MAX}"
        # Null check
        assert df["input"].notna().all(), f"{name}: null input"
        assert df["target"].notna().all(), f"{name}: null target"
        assert (df["input"].str.len() > 0).all(), f"{name}: empty input"
        assert (df["target"].str.len() > 0).all(), f"{name}: empty target"
        all_ids.extend(df["note_id"].tolist())

    # Duplicate check across splits
    assert len(all_ids) == len(set(all_ids)), "Duplicate note_ids across splits!"
    print("  ✓ All checks passed")


def save_jsonl(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Save each split as JSONL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        path = out_dir / f"{name}.jsonl"
        records = df.rename(
            columns={
                "input_tokens": "input_tokens_gpt4",
                "target_tokens": "target_tokens_gpt4",
            }
        ).to_dict(orient="records")
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {path.name}: {len(records):,} records")


def save_report(
    raw_count: int,
    after_range: dict[str, int],
    after_filter: dict[str, int],
    after_sample: dict[str, int],
    out_dir: Path,
) -> None:
    """Save preprocessing report."""
    report = {
        "raw_records": raw_count,
        "after_range_split": after_range,
        "after_target_filter": after_filter,
        "after_sample": after_sample,
        "sample_seed": SAMPLE_SEED,
        "target_filter": {"min": TARGET_MIN, "max": TARGET_MAX},
        "ranges": {k: {"lo": lo, "hi": hi} for k, (lo, hi) in RANGES.items()},
    }
    path = out_dir / "preprocessing_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {path}")


def step5_sample(
    splits: dict[str, pd.DataFrame],
    sample_size: int,
    seed: int = SAMPLE_SEED,
) -> dict[str, pd.DataFrame]:
    """Sample N records per range (stratified random sampling)."""
    sampled: dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        if sample_size >= len(df):
            sampled[name] = df
            print(f"  {name}: {len(df):,} records (kept all, < {sample_size})")
        else:
            sampled[name] = df.sample(n=sample_size, random_state=seed)
            print(f"  {name}: {len(df):,} → {sample_size} records (seed={seed})")
    return sampled


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV-BHC dataset")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Records per range (default: {DEFAULT_SAMPLE_SIZE}). Set to 0 to keep all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        sys.exit(1)

    # Load
    df = load_data(CSV_PATH)
    raw_count = len(df)

    # Step 1
    print("\n[Step 1] Splitting by context ranges ...")
    splits = step1_split_ranges(df)
    after_range = {k: len(v) for k, v in splits.items()}

    # Free raw df
    del df

    # Step 2
    print("\n[Step 2] Filtering target outliers ...")
    splits = step2_filter_target(splits)
    after_filter = {k: len(v) for k, v in splits.items()}

    # Steps 3 & 4: no-op (keep all sections + keep ___ tokens)
    print("\n[Step 3] Keeping all input sections intact — no changes")
    print("[Step 4] Keeping de-id tokens '___' — no changes")

    # Step 5: Sample
    if args.sample_size > 0:
        print(f"\n[Step 5] Sampling {args.sample_size} records per range ...")
        splits = step5_sample(splits, args.sample_size)
        after_sample = {k: len(v) for k, v in splits.items()}
    else:
        print("\n[Step 5] Sampling disabled — keeping all records")
        after_sample = after_filter

    # Validate
    print("\n[Validate] Running checks ...")
    validate(splits)

    # Save
    print("\n[Save] Writing JSONL files ...")
    save_jsonl(splits, OUT_DIR)
    save_report(raw_count, after_range, after_filter, after_sample, OUT_DIR)

    total = sum(after_sample.values())
    print(f"\n✓ Done! {total:,} records across 3 files in {OUT_DIR}")


if __name__ == "__main__":
    main()
