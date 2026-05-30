#!/usr/bin/env python3
"""Validate Unsloth-loaded Qwen3.5-4B against Ollama baseline.

Loads 5-10 samples, runs inference through both backends, and prints
a side-by-side quality comparison for manual review.

Usage:
    conda activate vinhthesis
    python scripts/validate_unsloth.py
    python scripts/validate_unsloth.py --max-samples 10
    python scripts/validate_unsloth.py --range 1k_2k
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from textwrap import shorten

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mimic_bhc import MIMICBHCLoader
from src.models.factory import ModelFactory
from src.prompts.templates import PromptTemplate
from src.utils.io import load_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────

UNSLOTH_MODEL_CONFIG = "configs/models/qwen3_5_4b_unsloth.yaml"
OLLAMA_PREDICTIONS_DIR = "outputs/baseline/qwen3_5_4b"
PROMPT_CONFIG = "configs/prompts/baseline.yaml"
DATA_DIR = "data/processed/mimic_iv_bhc"


# ── Helpers ───────────────────────────────────────────────────────────

def load_ollama_predictions(range_id: str) -> dict:
    """Load existing Ollama predictions for comparison."""
    pred_path = PROJECT_ROOT / OLLAMA_PREDICTIONS_DIR / f"range_{range_id}" / "predictions.jsonl"
    if not pred_path.exists():
        logger.warning(f"Ollama predictions not found: {pred_path}")
        return {}

    preds = {}
    with open(pred_path) as f:
        for line in f:
            record = json.loads(line.strip())
            preds[record["sample_id"]] = record.get("predicted_summary", "")
    return preds


def compare_outputs(
    sample_id: str,
    context: str,
    labeled: str,
    ollama_pred: str,
    unsloth_pred: str,
    unsloth_time: float,
) -> None:
    """Print side-by-side comparison for a single sample."""
    print(f"\n{'='*80}")
    print(f"  Sample: {sample_id}")
    print(f"  Context: {len(context)} chars | Ground truth: {len(labeled)} chars")
    print(f"  Unsloth inference time: {unsloth_time:.1f}s")
    print(f"{'='*80}")

    print(f"\n📋 GROUND TRUTH ({len(labeled)} chars):")
    print(f"  {shorten(labeled, width=300, placeholder='...')}")

    print(f"\n🟡 OLLAMA (Q8_0) ({len(ollama_pred)} chars):")
    print(f"  {shorten(ollama_pred, width=300, placeholder='...')}")

    print(f"\n🟢 UNSLOTH (4-bit NF4) ({len(unsloth_pred)} chars):")
    print(f"  {shorten(unsloth_pred, width=300, placeholder='...')}")

    # Basic length comparison
    ratio = len(unsloth_pred) / max(len(ollama_pred), 1)
    print(f"\n  Length ratio (Unsloth/Ollama): {ratio:.2f}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate Unsloth Qwen3.5-4B against Ollama baseline",
    )
    parser.add_argument(
        "--max-samples", type=int, default=5,
        help="Number of samples to compare (default: 5)",
    )
    parser.add_argument(
        "--range", type=str, default="0_1k",
        choices=["0_1k", "1k_2k", "2k_4k"],
        help="Token range to test (default: 0_1k)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save comparison results to JSONL file",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  Phase 4B: Unsloth Migration Validation")
    print("  Comparing Qwen3.5-4B: Ollama (Q8_0) vs Unsloth (4-bit NF4)")
    print("=" * 80)

    # Load data
    loader = MIMICBHCLoader(str(PROJECT_ROOT / DATA_DIR))
    samples = loader.load(range_id=args.range)
    samples = loader.sample(n=500, seed=42)  # Same seed as experiments
    samples = samples[:args.max_samples]
    logger.info(f"Loaded {len(samples)} samples from range_{args.range}")

    # Load existing Ollama predictions
    ollama_preds = load_ollama_predictions(args.range)
    logger.info(f"Loaded {len(ollama_preds)} Ollama predictions for comparison")

    # Load prompt template
    prompt_template = PromptTemplate.from_yaml(
        str(PROJECT_ROOT / PROMPT_CONFIG)
    )

    # Load Unsloth model
    model_config = load_yaml(str(PROJECT_ROOT / UNSLOTH_MODEL_CONFIG))
    logger.info("Loading Unsloth model...")
    model = ModelFactory.create(model_config)
    logger.info(f"Model info: {model.get_model_info()}")

    # Run inference and compare
    results = []
    total_time = 0.0

    for i, sample in enumerate(samples):
        prompt = prompt_template.format(context=sample.context)

        # Generate with Unsloth
        t0 = time.time()
        unsloth_pred = model.generate(prompt)
        elapsed = time.time() - t0
        total_time += elapsed

        # Get Ollama prediction (if available)
        ollama_pred = ollama_preds.get(sample.sample_id, "[No Ollama prediction found]")

        # Print comparison
        compare_outputs(
            sample_id=sample.sample_id,
            context=sample.context,
            labeled=sample.labeled_summary or "",
            ollama_pred=ollama_pred,
            unsloth_pred=unsloth_pred,
            unsloth_time=elapsed,
        )

        results.append({
            "sample_id": sample.sample_id,
            "context_len": len(sample.context),
            "labeled_len": len(sample.labeled_summary or ""),
            "ollama_pred": ollama_pred,
            "ollama_pred_len": len(ollama_pred),
            "unsloth_pred": unsloth_pred,
            "unsloth_pred_len": len(unsloth_pred),
            "unsloth_time_s": round(elapsed, 2),
        })

    # Summary
    avg_time = total_time / len(samples) if samples else 0
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Samples tested: {len(samples)}")
    print(f"  Average Unsloth inference time: {avg_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")

    # Length stats
    unsloth_lens = [r["unsloth_pred_len"] for r in results]
    ollama_lens = [r["ollama_pred_len"] for r in results if r["ollama_pred_len"] > 0]
    if unsloth_lens:
        print(f"  Avg Unsloth output length: {sum(unsloth_lens)/len(unsloth_lens):.0f} chars")
    if ollama_lens:
        print(f"  Avg Ollama output length: {sum(ollama_lens)/len(ollama_lens):.0f} chars")

    # Save if requested
    if args.save:
        save_path = PROJECT_ROOT / args.save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  Results saved to: {save_path}")

    # Cleanup
    model.cleanup()
    print(f"\n{'='*80}")
    print("  ✓ Validation complete. Review outputs above for quality.")
    print("  If quality is comparable, Unsloth migration is successful.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
