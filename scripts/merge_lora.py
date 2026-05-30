#!/usr/bin/env python3
"""Merge SFT LoRA adapter into full bf16 model for vLLM inference.

Uses Unsloth's save_pretrained_merged() to dequantize + merge QLoRA weights.

NOTE: Currently blocked on RTX 5060 Ti (Blackwell sm_120):
  - bitsandbytes requires libnvJitLink.so.13 (CUDA 13.x toolkit not installed)
  - Even load_in_4bit=False triggers it via Unsloth's VL model loading path
  See docs/known-issues.md for fix options.

Run in vinhthesis env (has Unsloth):
    conda activate vinhthesis
    python scripts/merge_lora.py

Output: models/qwen35_4b_sft_merged/ (bf16, ~8GB, vLLM-ready)
"""

import gc
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into full bf16 model.")
    parser.add_argument(
        "--adapter",
        type=str,
        default=str(PROJECT_ROOT / "models" / "qwen35_4b_sft_lora"),
        help="Path to saved LoRA adapter."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "qwen35_4b_sft_merged"),
        help="Output path for merged bf16 weights."
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    merged_path  = Path(args.output)

    logger.info("=" * 60)
    logger.info("Merging LoRA via Unsloth native merge")
    logger.info(f"  Adapter:  {adapter_path}")
    logger.info(f"  Output:   {merged_path}")
    logger.info("=" * 60)

    # ── Step 1: Load via PEFT / AutoModelForCausalLM in bf16 ──────────
    logger.info("Step 1/2: Loading base model + PEFT adapter in bf16...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # ── Step 2: Merge and save in bf16 ────────────────────────────────
    logger.info("Step 2/2: Merging and saving merged bf16 model...")
    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(merged_path))

    # ── Report ────────────────────────────────────────────────────────
    total_bytes = sum(f.stat().st_size for f in merged_path.glob("*.safetensors"))
    logger.info(f"  Saved to: {merged_path}")
    logger.info(f"  Model size: {total_bytes / 1e9:.2f} GB")

    del merged_model, base_model, tokenizer
    gc.collect()

    logger.info("✓ Merge complete!")



if __name__ == "__main__":
    main()

