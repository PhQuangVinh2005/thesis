#!/usr/bin/env python3
"""Extract SFT training data from MIMIC-IV-BHC.

Randomly samples N records from the full dataset, EXCLUDING the
1,500 test samples used for evaluation. Applies target outlier
filters and max token length filtering to ensure all samples fit
within the model's max_seq_length.

Usage:
    python scripts/extract_sft_data.py --n-samples 30000
    python scripts/extract_sft_data.py --n-samples 30000 --max-total-tokens 3800

Output:
    data/processed/sft/train_30k.jsonl
    data/processed/sft/extraction_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────

RAW_CSV = (
    PROJECT_ROOT / "data" / "raw"
    / "mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0"
    / "mimic-iv-bhc.csv"
)
TEST_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mimic_iv_bhc"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sft"

# ── Constants ─────────────────────────────────────────────────────────

TEST_RANGES = ["0_1k", "1k_2k", "2k_4k"]
TARGET_MIN_TOKENS = 50
TARGET_MAX_TOKENS = 2000
SEED = 42


def get_test_ids() -> set:
    """Load note_ids from the 1,500 test samples to exclude."""
    test_ids = set()
    for range_id in TEST_RANGES:
        filepath = TEST_DATA_DIR / f"range_{range_id}.jsonl"
        if not filepath.exists():
            logger.warning(f"Test file not found: {filepath}")
            continue
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    test_ids.add(rec["note_id"])
    logger.info(f"Test IDs to exclude: {len(test_ids)}")
    return test_ids


def validate_output(out_path: Path, expected_count: int) -> bool:
    """Validate written JSONL file integrity."""
    logger.info("Validating output file...")
    actual = 0
    errors = 0
    with open(out_path) as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
                # Check required fields
                assert "note_id" in rec, f"Line {i}: missing note_id"
                assert "input" in rec, f"Line {i}: missing input"
                assert "target" in rec, f"Line {i}: missing target"
                assert len(rec["input"]) > 0, f"Line {i}: empty input"
                assert len(rec["target"]) > 0, f"Line {i}: empty target"
                actual += 1
            except (json.JSONDecodeError, AssertionError) as e:
                logger.error(f"Validation error: {e}")
                errors += 1

    if actual != expected_count:
        logger.error(f"Count mismatch: expected {expected_count}, got {actual}")
        return False
    if errors > 0:
        logger.error(f"Found {errors} corrupted records")
        return False

    logger.info(f"✓ Validation passed: {actual:,} records, 0 errors")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract SFT training data")
    parser.add_argument(
        "--n-samples", type=int, default=30000,
        help="Number of training samples to extract (default: 30000)",
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=3800,
        help="Max input+target GPT4 tokens (default: 3800, leaves room for overhead in 4096 seq)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    args = parser.parse_args()

    if not RAW_CSV.exists():
        logger.error(f"Raw dataset not found: {RAW_CSV}")
        sys.exit(1)

    # ── Load full dataset ─────────────────────────────────────────────
    logger.info(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    logger.info(f"Loaded {len(df):,} records")

    # ── Exclude test set ──────────────────────────────────────────────
    test_ids = get_test_ids()
    before = len(df)
    df = df[~df["note_id"].isin(test_ids)]
    logger.info(f"Excluded {before - len(df)} test samples → {len(df):,} remaining")

    # ── Filter target outliers ────────────────────────────────────────
    before = len(df)
    df = df[
        (df["target_tokens"] >= TARGET_MIN_TOKENS)
        & (df["target_tokens"] <= TARGET_MAX_TOKENS)
    ]
    logger.info(f"Target filter ({TARGET_MIN_TOKENS}-{TARGET_MAX_TOKENS}) → {len(df):,} (removed {before - len(df)})")

    # ── Filter by max total tokens ────────────────────────────────────
    df["total_tokens"] = df["input_tokens"] + df["target_tokens"]
    before = len(df)
    df = df[df["total_tokens"] <= args.max_total_tokens]
    logger.info(f"Max total tokens ≤ {args.max_total_tokens} → {len(df):,} (removed {before - len(df)})")

    # ── Drop rows with missing text ───────────────────────────────────
    df = df.dropna(subset=["input", "target"])
    df = df[df["input"].str.len() > 0]
    df = df[df["target"].str.len() > 0]
    logger.info(f"After null/empty filter → {len(df):,}")

    # ── Sample ────────────────────────────────────────────────────────
    n = min(args.n_samples, len(df))
    if n < args.n_samples:
        logger.warning(f"Only {len(df):,} eligible samples, using all (requested {args.n_samples})")

    df_sampled = df.sample(n=n, random_state=args.seed)
    logger.info(f"Sampled {len(df_sampled):,} records (seed={args.seed})")

    # ── Stats ─────────────────────────────────────────────────────────
    logger.info("=== Sample Statistics ===")
    logger.info(f"  Input tokens  — mean: {df_sampled['input_tokens'].mean():.0f}, "
                f"median: {df_sampled['input_tokens'].median():.0f}, "
                f"max: {df_sampled['input_tokens'].max()}")
    logger.info(f"  Target tokens — mean: {df_sampled['target_tokens'].mean():.0f}, "
                f"median: {df_sampled['target_tokens'].median():.0f}, "
                f"max: {df_sampled['target_tokens'].max()}")
    logger.info(f"  Total tokens  — mean: {df_sampled['total_tokens'].mean():.0f}, "
                f"median: {df_sampled['total_tokens'].median():.0f}, "
                f"max: {df_sampled['total_tokens'].max()}")

    # Range distribution
    bins = [0, 1024, 2048, 4096, 99999]
    labels = ["0-1K", "1K-2K", "2K-4K", "4K+"]
    df_sampled["range"] = pd.cut(df_sampled["input_tokens"], bins=bins, labels=labels, right=True)
    range_dist = df_sampled["range"].value_counts().sort_index()
    logger.info("  Range distribution:")
    for rng, count in range_dist.items():
        logger.info(f"    {rng}: {count:,} ({100*count/len(df_sampled):.1f}%)")

    # ── Save JSONL with tqdm ──────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"train_{n // 1000}k.jsonl"

    records = df_sampled[["note_id", "input", "target", "input_tokens", "target_tokens"]].rename(
        columns={"input_tokens": "input_tokens_gpt4", "target_tokens": "target_tokens_gpt4"}
    ).to_dict(orient="records")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in tqdm(records, desc="Writing JSONL", unit="samples"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved → {out_path}")

    # ── Validate output ───────────────────────────────────────────────
    if not validate_output(out_path, n):
        logger.error("Output validation FAILED! Check the file.")
        sys.exit(1)

    # ── Save report ───────────────────────────────────────────────────
    report = {
        "n_samples": n,
        "seed": args.seed,
        "max_total_tokens": args.max_total_tokens,
        "target_filter": {"min": TARGET_MIN_TOKENS, "max": TARGET_MAX_TOKENS},
        "test_ids_excluded": len(test_ids),
        "input_tokens_mean": round(df_sampled["input_tokens"].mean(), 1),
        "input_tokens_median": round(df_sampled["input_tokens"].median(), 1),
        "target_tokens_mean": round(df_sampled["target_tokens"].mean(), 1),
        "target_tokens_median": round(df_sampled["target_tokens"].median(), 1),
        "total_tokens_mean": round(df_sampled["total_tokens"].mean(), 1),
        "total_tokens_max": int(df_sampled["total_tokens"].max()),
        "range_distribution": {k: int(v) for k, v in range_dist.items()},
        "output_file": str(out_path),
    }
    report_path = OUT_DIR / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report → {report_path}")

    logger.info(f"✓ Done! {n:,} SFT samples ready at {out_path}")


if __name__ == "__main__":
    main()
