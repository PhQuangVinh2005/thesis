#!/usr/bin/env python3
"""
Cross-Tokenizer EDA: So sánh token count giữa GPT-4 (dataset) vs BioMistral / Qwen3.5.

Mục đích:
  - Xem tokenizer của BioMistral và Qwen3.5 tokenize text khác GPT-4 bao nhiêu
  - Xác định max_model_len hợp lý cho từng model
  - Quyết định có cần chia lại range hay không

Input:  data/processed/mimic_iv_bhc/range_*.jsonl  (1500 samples)
Output:
  - Console: bảng thống kê + khuyến nghị
  - notebooks/EDA/02_cross_tokenizer_stats.csv
  - notebooks/EDA/02_cross_tokenizer_plots.png

Usage:
    conda activate vinhthesis
    python notebooks/EDA/02_cross_tokenizer_eda.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mimic_iv_bhc"
OUTPUT_DIR = Path(__file__).resolve().parent  # notebooks/EDA/

RANGES = ["range_0_1k", "range_1k_2k", "range_2k_4k"]

# ── Model specs (from HuggingFace model cards) ─────────────────────────
MODEL_SPECS = {
    "biomistral_7b": {
        "hf_name": "BioMistral/BioMistral-7B",
        "max_context": 32768,       # max_position_embeddings
        "sliding_window": 4096,     # effective attention window
        "vocab_size": 32000,
    },
    "qwen3_5": {
        "hf_name": "Qwen/Qwen3.5-2B",
        "max_context": 262144,      # native context length
        "sliding_window": None,     # no sliding window
        "vocab_size": 248320,       # padded
    },
}


def load_preprocessed_data() -> pd.DataFrame:
    """Load all preprocessed JSONL files into one DataFrame."""
    frames = []
    for range_name in RANGES:
        path = DATA_DIR / f"{range_name}.jsonl"
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["range"] = range_name
                records.append(rec)
        frames.append(pd.DataFrame(records))
        print(f"  Loaded {len(records)} records from {path.name}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(df)} records\n")
    return df


def load_tokenizers() -> dict:
    """Load tokenizers (NOT model weights) from HuggingFace."""
    from transformers import AutoTokenizer

    tokenizers = {}
    for name, spec in MODEL_SPECS.items():
        print(f"  Loading tokenizer: {spec['hf_name']} ...")
        tokenizers[name] = AutoTokenizer.from_pretrained(
            spec["hf_name"], trust_remote_code=True
        )
        print(f"    vocab_size={tokenizers[name].vocab_size}")
    print()
    return tokenizers


def tokenize_column(df: pd.DataFrame, col: str, tokenizer, name: str) -> pd.Series:
    """Tokenize a text column and return token counts."""
    col_name = f"{col}_tokens_{name}"
    print(f"  Tokenizing '{col}' with {name} ... ", end="", flush=True)
    counts = df[col].apply(lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    print(f"done (mean={counts.mean():.0f}, max={counts.max()})")
    return counts


def compute_stats(df: pd.DataFrame, tokenizer_names: list[str]) -> pd.DataFrame:
    """Compute per-range per-tokenizer statistics."""
    rows = []
    for range_name in RANGES:
        sub = df[df["range"] == range_name]
        for tok_name in ["gpt4"] + tokenizer_names:
            for field in ["input", "target"]:
                col = f"{field}_tokens_{tok_name}"
                if col not in sub.columns:
                    continue
                vals = sub[col]
                row = {
                    "range": range_name,
                    "tokenizer": tok_name,
                    "field": field,
                    "count": len(vals),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "min": vals.min(),
                    "p50": vals.median(),
                    "p95": vals.quantile(0.95),
                    "p99": vals.quantile(0.99),
                    "max": vals.max(),
                }
                # Ratio vs GPT-4
                gpt4_col = f"{field}_tokens_gpt4"
                if tok_name != "gpt4" and gpt4_col in sub.columns:
                    ratios = sub[col] / sub[gpt4_col]
                    row["ratio_mean"] = ratios.mean()
                    row["ratio_std"] = ratios.std()
                rows.append(row)
    return pd.DataFrame(rows)


def print_recommendations(df: pd.DataFrame, tokenizer_names: list[str]):
    """Print recommended max_model_len based on p99 of input tokens."""
    print("\n" + "=" * 70)
    print("RECOMMENDED max_model_len (based on p99 of input tokens + 1024 output)")
    print("=" * 70)

    for tok_name in tokenizer_names:
        spec = MODEL_SPECS[tok_name]
        print(f"\n  {tok_name} ({spec['hf_name']}):")
        print(f"    Architecture max_context: {spec['max_context']:,}")
        if spec["sliding_window"]:
            print(f"    Sliding window: {spec['sliding_window']:,}")

        for range_name in RANGES:
            sub = df[df["range"] == range_name]
            input_col = f"input_tokens_{tok_name}"
            if input_col not in sub.columns:
                continue
            p99 = sub[input_col].quantile(0.99)
            recommended = int(np.ceil((p99 + 1024) / 256) * 256)  # round up to 256
            effective_max = spec["sliding_window"] or spec["max_context"]
            fits = "✅" if recommended <= effective_max else "⚠️ EXCEEDS"
            print(f"    {range_name}: p99_input={p99:.0f} → recommended={recommended:,} {fits}")


def make_plots(df: pd.DataFrame, tokenizer_names: list[str], output_path: Path):
    """Create comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not available, skipping plots")
        return

    sns.set_theme(style="whitegrid", font_scale=1.0)

    n_tok = len(tokenizer_names)
    fig, axes = plt.subplots(n_tok, 3, figsize=(18, 5 * n_tok))
    if n_tok == 1:
        axes = axes.reshape(1, -1)

    for i, tok_name in enumerate(tokenizer_names):
        for j, range_name in enumerate(RANGES):
            ax = axes[i, j]
            sub = df[df["range"] == range_name]
            gpt4_col = "input_tokens_gpt4"
            model_col = f"input_tokens_{tok_name}"

            if model_col not in sub.columns:
                ax.set_visible(False)
                continue

            ax.scatter(sub[gpt4_col], sub[model_col], alpha=0.4, s=10, c="steelblue")
            # y=x line
            lims = [
                min(sub[gpt4_col].min(), sub[model_col].min()),
                max(sub[gpt4_col].max(), sub[model_col].max()),
            ]
            ax.plot(lims, lims, "r--", alpha=0.5, label="y=x")

            ratio = (sub[model_col] / sub[gpt4_col]).mean()
            ax.set_title(f"{tok_name} | {range_name}\nmean ratio = {ratio:.3f}")
            ax.set_xlabel("GPT-4 tokens")
            ax.set_ylabel(f"{tok_name} tokens")
            ax.legend()

    plt.suptitle("Cross-Tokenizer Comparison: Input Token Counts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plots saved to: {output_path}")


def main():
    print("=" * 70)
    print("Cross-Tokenizer EDA: GPT-4 vs BioMistral vs Qwen3.5")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading preprocessed data ...")
    df = load_preprocessed_data()

    # 2. Load tokenizers
    print("[2] Loading tokenizers ...")
    tokenizers = load_tokenizers()

    # 3. Rename existing GPT-4 columns for consistency
    # (already named input_tokens_gpt4, target_tokens_gpt4 from preprocessing)

    # 4. Tokenize with each model's tokenizer
    print("[3] Tokenizing with model tokenizers ...")
    tokenizer_names = list(tokenizers.keys())
    for tok_name, tokenizer in tokenizers.items():
        df[f"input_tokens_{tok_name}"] = tokenize_column(df, "input", tokenizer, tok_name)
        df[f"target_tokens_{tok_name}"] = tokenize_column(df, "target", tokenizer, tok_name)
    print()

    # 5. Compute statistics
    print("[4] Computing statistics ...")
    stats = compute_stats(df, tokenizer_names)

    # Print summary
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}".format)

    print("\n--- INPUT token statistics ---")
    input_stats = stats[stats["field"] == "input"].copy()
    print(input_stats.to_string(index=False))

    print("\n--- TARGET token statistics ---")
    target_stats = stats[stats["field"] == "target"].copy()
    print(target_stats.to_string(index=False))

    # 6. Save CSV
    csv_path = OUTPUT_DIR / "02_cross_tokenizer_stats.csv"
    stats.to_csv(csv_path, index=False)
    print(f"\n  Stats saved to: {csv_path}")

    # 7. Recommendations
    print_recommendations(df, tokenizer_names)

    # 8. Plots
    print("\n[5] Generating plots ...")
    plot_path = OUTPUT_DIR / "02_cross_tokenizer_plots.png"
    make_plots(df, tokenizer_names, plot_path)

    # 9. Print model context info
    print("\n" + "=" * 70)
    print("MODEL CONTEXT WINDOW REFERENCE")
    print("=" * 70)
    for name, spec in MODEL_SPECS.items():
        print(f"\n  {name} ({spec['hf_name']}):")
        print(f"    max_position_embeddings: {spec['max_context']:,}")
        if spec["sliding_window"]:
            print(f"    sliding_window: {spec['sliding_window']:,}")
            print(f"    → Effective context = {spec['sliding_window']:,} tokens")
        else:
            print(f"    sliding_window: None")
            print(f"    → Effective context = {spec['max_context']:,} tokens")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
