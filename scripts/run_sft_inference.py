#!/usr/bin/env python3
"""Run SFT model inference on 1,500 test samples using vLLM.

Produces predictions.jsonl files compatible with existing run_evaluation.py.

Prerequisites:
    1. Fix merged model config (in vllm env):
       conda activate vllm
       python scripts/fix_merged_config.py

    2. Run inference (in vllm env):
       conda activate vllm
       python scripts/run_sft_inference.py

    3. Evaluate (in appropriate env):
       python scripts/run_evaluation.py \\
           --experiment-dir outputs/baseline/qwen3_5_4b_sft/ \\
           --phase completeness

Options:
    --dry-run         Print prompts, don't generate
    --max-samples N   Limit samples per range (debug)
    --range RANGE     Single range only (e.g., 0_1k)
    --temperature T   Sampling temperature (default: 0.1)
    --max-tokens N    Max new tokens (default: 1024)
"""

import os

# ── vLLM nightly (0.21.1+cu130) — Blackwell sm_120 compatibility ─────
# FlashInfer's JIT sampler fails the sm75 arch check on Blackwell.
# Disable → falls back to PyTorch-native sampling.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

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
MERGED_MODEL = PROJECT_ROOT / "models" / "qwen35_4b_sft_merged"
DATA_DIR     = PROJECT_ROOT / "data" / "processed" / "mimic_iv_bhc"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "baseline" / "qwen3_5_4b_sft"
RANGES       = ["0_1k", "1k_2k", "2k_4k"]

# ── Generation limits ─────────────────────────────────────────────────
MAX_MODEL_LEN  = 6144   # vLLM context window (fits 2k-4k GPT4-token notes)
MAX_GEN_TOKENS = 1024   # max new tokens per sample
MAX_INPUT_TOKENS = MAX_MODEL_LEN - MAX_GEN_TOKENS  # 5120 tokens for prompt

# ── Prompt (matches configs/prompts/baseline.yaml) ────────────────────
SYSTEM_PROMPT = ""
INSTRUCTION = """You are an expert medical professional specializing in clinical documentation.

Summarize the following clinical notes into a concise Brief Hospital Course. Focus only on clinically significant events, diagnoses, and treatments. Ensure the summary is technically accurate and uses professional medical terminology.

Strictly adhere to the provided context. If a piece of information is not explicitly stated in the input text, do not include it. Avoid inferring results or patient outcomes.

input: {context}
summary:"""


def load_test_samples(range_id: str) -> list[dict]:
    """Load samples from range_*.jsonl."""
    filepath = DATA_DIR / f"range_{range_id}.jsonl"
    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


_tokenizer = None


def _get_tokenizer():
    """Lazy-load tokenizer for prompt truncation."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            str(MERGED_MODEL), trust_remote_code=True
        )
    return _tokenizer


def build_prompts(samples: list[dict]) -> list[str]:
    """Build prompts, truncating inputs that exceed MAX_INPUT_TOKENS."""
    tokenizer = _get_tokenizer()
    prompts = []
    truncated = 0

    # Tokenize instruction template (without context) to measure overhead
    tmpl_tokens = len(tokenizer.encode(
        INSTRUCTION.replace("{context}", ""), add_special_tokens=False
    ))
    max_context_tokens = MAX_INPUT_TOKENS - tmpl_tokens

    for s in samples:
        context = s["input"]
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)

        if len(ctx_ids) > max_context_tokens:
            ctx_ids = ctx_ids[:max_context_tokens]
            context = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            truncated += 1

        prompts.append(INSTRUCTION.format(context=context))

    if truncated:
        logger.info(f"Truncated {truncated}/{len(samples)} prompts to fit {MAX_INPUT_TOKENS} input tokens")

    return prompts


SAVE_EVERY = 50  # auto-save after every N samples


def _save_records(pred_file: Path, records: list[dict]) -> None:
    """Atomically write all records to predictions.jsonl."""
    tmp = pred_file.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.rename(pred_file)  # atomic on same filesystem


def run_range(
    llm,
    sampling_params,
    range_id: str,
    args,
) -> None:
    """Run inference for one range with auto-save every SAVE_EVERY samples."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Range: {range_id}")
    logger.info(f"{'='*60}")

    samples = load_test_samples(range_id)
    logger.info(f"Loaded {len(samples)} samples")

    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"Limited to {len(samples)} samples (--max-samples)")

    prompts = build_prompts(samples)

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        for i, (s, p) in enumerate(zip(samples[:3], prompts[:3])):
            print(f"\n--- Sample {i+1}: {s['note_id']} ---")
            print(f"Input: {len(s['input'])} chars")
            print(f"Prompt (first 300): {p[:300]}...")
        logger.info(f"[DRY RUN] {len(samples)} samples would be processed")
        return

    # ── Resume support ────────────────────────────────────────────────
    output_dir = OUTPUT_DIR / f"range_{range_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "predictions.jsonl"

    completed_ids = set()
    all_records = []
    if pred_file.exists():
        with open(pred_file) as f:
            for line in f:
                record = json.loads(line)
                if record.get("predicted_summary"):
                    completed_ids.add(record["sample_id"])
                    all_records.append(record)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} already done")

    # Filter pending
    pending_indices = [i for i, s in enumerate(samples) if s["note_id"] not in completed_ids]
    pending_samples = [samples[i] for i in pending_indices]
    pending_prompts = [prompts[i] for i in pending_indices]

    if not pending_prompts:
        logger.info("All samples already completed!")
        return

    logger.info(f"Generating {len(pending_prompts)} summaries (auto-save every {SAVE_EVERY})...")

    # ── Chunked inference with auto-save ──────────────────────────────
    total_gen_tokens = 0
    total_errors = 0
    t0 = time.time()

    for chunk_start in range(0, len(pending_prompts), SAVE_EVERY):
        chunk_end = min(chunk_start + SAVE_EVERY, len(pending_prompts))
        chunk_prompts = pending_prompts[chunk_start:chunk_end]
        chunk_samples = pending_samples[chunk_start:chunk_end]

        chunk_t0 = time.time()
        outputs = llm.generate(chunk_prompts, sampling_params)
        chunk_elapsed = time.time() - chunk_t0

        chunk_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_gen_tokens += chunk_tokens

        for sample, output in zip(chunk_samples, outputs):
            text = output.outputs[0].text.strip()
            if not text:
                total_errors += 1
                text = "[ERROR] Empty generation"

            record = {
                "sample_id": sample["note_id"],
                "context": sample["input"],
                "predicted_summary": text,
                "labeled_summary": sample["target"],
                "instruction": INSTRUCTION.replace("{context}", "..."),
                "inference_time_s": round(chunk_elapsed / len(outputs), 2),
            }
            all_records.append(record)

        # Auto-save after each chunk
        _save_records(pred_file, all_records)
        done = len(completed_ids) + chunk_end
        total = len(completed_ids) + len(pending_prompts)
        tps = chunk_tokens / chunk_elapsed if chunk_elapsed > 0 else 0
        logger.info(
            f"  Saved {done}/{total} | chunk {chunk_elapsed:.1f}s "
            f"({tps:.0f} tok/s)"
        )

    elapsed = time.time() - t0
    logger.info(f"Generation: {elapsed:.1f}s total, {elapsed/len(pending_prompts):.1f}s/sample")
    if total_gen_tokens > 0 and elapsed > 0:
        logger.info(f"Throughput: {total_gen_tokens/elapsed:.1f} tokens/sec")

    # ── Save experiment metadata ──────────────────────────────────────
    meta = {
        "experiment_name": "qwen3_5_4b_sft",
        "model": {
            "model_name": "Qwen/Qwen3.5-4B",
            "backend": "vllm",
            "lora_adapter": "models/qwen35_4b_sft_lora/",
            "merged_model": str(MERGED_MODEL),
        },
        "prompt": {"name": "baseline_summarize"},
        "data": {
            "range": range_id,
            "total_loaded": len(samples),
            "actual_processed": len(all_records),
        },
        "performance": {
            "total_time_s": round(elapsed, 1),
            "avg_time_per_sample_s": round(elapsed / len(pending_prompts), 2),
            "tokens_per_sec": round(total_gen_tokens / elapsed, 1) if elapsed > 0 else 0,
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved {len(all_records)} predictions to {pred_file}")
    if total_errors:
        logger.warning(f"{total_errors} empty generations")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT inference via vLLM")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit per range")
    parser.add_argument("--range", type=str, default=None, help="Single range (e.g., 0_1k)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=MAX_GEN_TOKENS)
    parser.add_argument("--model", type=str, default=None,
                        help="Override model path (default: models/qwen35_4b_sft_merged)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: outputs/baseline/qwen3_5_4b_sft)")
    return parser.parse_args()


def main():
    global OUTPUT_DIR
    args = parse_args()

    model_path = args.model or str(MERGED_MODEL)
    ranges = [args.range] if args.range else RANGES

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("SFT Inference via vLLM")
    logger.info(f"  Model:  {model_path}")
    logger.info(f"  Ranges: {ranges}")
    logger.info(f"  Temp:   {args.temperature}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Output: {OUTPUT_DIR}")
    logger.info("=" * 60)

    if not args.dry_run:
        # Check model exists
        if not Path(model_path).exists():
            logger.error(
                f"Merged model not found: {model_path}\n"
                f"Run first: conda activate vllm && python scripts/fix_merged_config.py"
            )
            sys.exit(1)

        from vllm import LLM, SamplingParams

        logger.info("Loading vLLM engine...")
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.50,
            enforce_eager=True,       # skip torch.compile + CUDA graphs (saves ~2GB)
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_tokens,
            repetition_penalty=1.2,   # prevent degenerate repetition loops
        )
    else:
        llm = None
        sampling_params = None

    for range_id in ranges:
        run_range(llm, sampling_params, range_id, args)

    logger.info("\n✓ All inference complete!")
    logger.info("Next: evaluate with run_evaluation.py")


if __name__ == "__main__":
    main()
