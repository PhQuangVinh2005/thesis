#!/usr/bin/env python3
"""SFT training for Qwen3.5-4B on MIMIC-IV-BHC (Phase 4C).

Domain adaptation via supervised finetuning on pre-extracted training data
from scripts/extract_sft_data.py.

Prerequisites:
    python scripts/extract_sft_data.py --n-samples 30000

Usage:
    conda activate vinhthesis
    python scripts/train_sft.py --dry-run          # Check data only
    python scripts/train_sft.py --dry-run-format    # Check data + model + formatting
    python scripts/train_sft.py                     # Full training (~30-50 hrs)

Output:
    models/qwen35_4b_sft_lora/   (LoRA adapter weights)
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# ── VRAM fragmentation fix (must be set BEFORE torch import) ──────────
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP/hub loggers (hundreds of 404 probes during model download)
for _noisy in ("httpx", "urllib3", "huggingface_hub", "httpcore", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "sft" / "train_30k.jsonl"

SYSTEM_PROMPT = (
    "You are an expert medical professional specializing in clinical documentation. "
    "Summarize the following clinical notes into a concise Brief Hospital Course. "
    "Focus only on clinically significant events, diagnoses, and treatments. "
    "Ensure the summary is technically accurate and uses professional medical terminology. "
    "Strictly adhere to the provided context. If a piece of information is not explicitly "
    "stated in the input text, do not include it. Avoid inferring results or patient outcomes."
)

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen35_4b_sft_lora"


# ── Data Preparation ─────────────────────────────────────────────────

def load_sft_data(data_path: Path) -> list:
    """Load pre-extracted SFT JSONL data.

    Expected fields: note_id, input, target, input_tokens_gpt4, target_tokens_gpt4
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"SFT data not found: {data_path}\n"
            "Run: python scripts/extract_sft_data.py --n-samples 30000"
        )

    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records):,} SFT samples from {data_path.name}")

    # Quick sanity checks
    bad = sum(1 for r in records if not r.get("input") or not r.get("target"))
    if bad > 0:
        logger.error(f"Found {bad} records with empty input/target!")
        raise ValueError(f"Data integrity issue: {bad} bad records")

    return records


def format_for_sft(records: list, tokenizer) -> list:
    """Convert records to chat-formatted text for SFT training.

    Each sample becomes a full conversation:
        system → user (clinical notes) → assistant (ground truth BHC)

    Returns list of dicts with 'text' key containing the formatted conversation.
    """
    formatted = []
    for rec in tqdm(records, desc="Formatting chat templates", unit="samples"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rec["input"]},
            {"role": "assistant", "content": rec["target"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        formatted.append({"text": text})

    logger.info(f"Formatted {len(formatted):,} samples for SFT")

    # Log sample lengths for debugging
    lengths = [len(f["text"]) for f in formatted]
    logger.info(
        f"Text char lengths — min: {min(lengths)}, max: {max(lengths)}, "
        f"avg: {sum(lengths)/len(lengths):.0f}"
    )
    return formatted


# ── Training ──────────────────────────────────────────────────────────

def train(args):
    """Run SFT training."""

    # ── Load data first (fail fast on data issues) ────────────────────
    data_path = Path(args.data) if args.data else DEFAULT_DATA
    records = load_sft_data(data_path)

    if args.dry_run:
        # Quick dry run: just check data, no model loading
        logger.info(f"[DRY RUN] {len(records):,} samples loaded")
        input_tokens = [r.get("input_tokens_gpt4", 0) for r in records]
        target_tokens = [r.get("target_tokens_gpt4", 0) for r in records]
        total = [i + t for i, t in zip(input_tokens, target_tokens)]
        logger.info(f"  Input tokens  — mean: {sum(input_tokens)/len(input_tokens):.0f}, max: {max(input_tokens)}")
        logger.info(f"  Target tokens — mean: {sum(target_tokens)/len(target_tokens):.0f}, max: {max(target_tokens)}")
        logger.info(f"  Total tokens  — mean: {sum(total)/len(total):.0f}, max: {max(total)}")
        logger.info(f"  Max seq length setting: {args.max_seq_length}")
        over = sum(1 for t in total if t + 200 > args.max_seq_length)
        logger.info(f"  Samples exceeding max_seq_length: {over} ({100*over/len(total):.1f}%)")
        return

    # ── Heavy imports (AFTER dry-run exit, Unsloth MUST come first) ────
    from unsloth import FastLanguageModel
    import torch

    # ── cuBLAS Handle Warm-up ─────────────────────────────────────────
    # Initialize cuBLAS early under 100% free VRAM to prevent CUBLAS_STATUS_ALLOC_FAILED during training
    if torch.cuda.is_available():
        logger.info("Pre-initializing cuBLAS handle to prevent allocation failure...")
        try:
            _temp_a = torch.randn(1, 1, device="cuda")
            _temp_b = torch.matmul(_temp_a, _temp_a)
            del _temp_a, _temp_b
            torch.cuda.empty_cache()
            logger.info("cuBLAS handle successfully initialized.")
        except Exception as e:
            logger.warning(f"Failed to pre-initialize cuBLAS handle: {e}")

    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    # Define callback here so it inherits from TrainerCallback after import
    class CSVLoggingCallback(TrainerCallback):
        """Write step-level metrics to CSV. Flushed per-step for crash safety."""

        def __init__(self, csv_path: Path, resume: bool = False):
            self.csv_path = csv_path
            self._file = None
            self._writer = None
            self._resume = resume

        def on_train_begin(self, args, state, control, **kwargs):
            if self._resume and self.csv_path.exists():
                # Append to existing CSV on resume
                self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
                logger.info(f"Appending to existing CSV → {self.csv_path}")
            else:
                self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
                self._writer = csv.writer(self._file)
                self._writer.writerow(["step", "epoch", "train_loss", "eval_loss", "learning_rate", "timestamp"])
                logger.info(f"Training log CSV → {self.csv_path}")
            self._writer = csv.writer(self._file)
            self._file.flush()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or self._writer is None:
                return
            row = [
                state.global_step,
                f"{state.epoch:.2f}" if state.epoch else "",
                logs.get("loss", ""),
                logs.get("eval_loss", ""),
                logs.get("learning_rate", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ]
            self._writer.writerow(row)
            self._file.flush()

        def on_train_end(self, args, state, control, **kwargs):
            if self._file:
                self._file.close()
                logger.info(f"Training log CSV closed: {self.csv_path}")

    # ── Load model ────────────────────────────────────────────────────
    logger.info(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # ── Attach LoRA adapters ──────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    # ── VRAM safety check + cleanup ────────────────────────────────────
    if torch.cuda.is_available():
        # Clear fragmented VRAM before training
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc; gc.collect()

        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        vram_free = vram_total - vram_used
        logger.info(f"VRAM: {vram_used:.1f}GB used / {vram_total:.1f}GB total ({vram_free:.1f}GB free)")
        if vram_free < 4.0 and args.batch_size > 1:
            logger.warning(
                f"Low VRAM ({vram_free:.1f}GB free) with batch_size={args.batch_size}. "
                f"Consider --batch-size 1 --grad-accum 16 to avoid OOM."
            )

    # ── Format data ───────────────────────────────────────────────────
    formatted = format_for_sft(records, tokenizer)
    del records  # Free memory

    if args.dry_run_format:
        # Dry run with formatting: show a sample
        logger.info("[DRY RUN] Sample formatted text (first 500 chars):")
        print(formatted[0]["text"][:500] + "...")
        logger.info(f"[DRY RUN] Total formatted: {len(formatted):,} samples. All good!")
        return

    dataset = Dataset.from_list(formatted)
    del formatted  # Free memory

    # Split: 95% train, 5% eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")

    # ── Get text tokenizer for SFT ────────────────────────────────────
    # Unsloth returns a VL processor for Qwen3.5; extract underlying tokenizer
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # ── Compute save steps ────────────────────────────────────────────
    # Save frequently to survive OOM crashes (every ~50 optimizer steps)
    steps_per_epoch = len(train_dataset) // args.batch_size
    optimizer_steps_per_epoch = steps_per_epoch // args.grad_accum
    save_steps = args.save_steps  # Use CLI arg (default: 50)
    logger.info(
        f"Steps per epoch: {steps_per_epoch:,} micro-batches, "
        f"{optimizer_steps_per_epoch:,} optimizer steps. "
        f"Saving every {save_steps} optimizer steps "
        f"(= every {save_steps * args.grad_accum} micro-batches)."
    )

    # ── Training config ───────────────────────────────────────────────
    output_dir = str(args.output_dir or OUTPUT_DIR)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,  # Must match train — default 8 causes OOM
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,         # Regularization against overfitting
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,         # 5% warmup (shorter for large dataset)
        bf16=True,
        logging_steps=10,          # Log every 10 steps for visibility
        save_strategy="steps",
        save_steps=save_steps,     # Checkpoint frequently to survive crashes
        eval_strategy="steps",
        eval_steps=save_steps,     # Eval at same frequency as save
        save_total_limit=5,        # Keep last 5 checkpoints (OOM safety)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,             # No packing — each sample is one conversation
        seed=42,
        report_to="none",          # No wandb for thesis
    )

    # ── File-based logging (survives SSH disconnects) ────────────────
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.resume else "w"  # Append on resume
    file_handler = logging.FileHandler(log_dir / "training.log", mode=log_mode)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"File logging → {log_dir / 'training.log'} (mode={log_mode})")

    # ── CSV metrics callback ──────────────────────────────────────────
    csv_callback = CSVLoggingCallback(csv_path=log_dir / "training_metrics.csv", resume=args.resume)

    # ── Train ─────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=text_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[csv_callback],
    )

    logger.info("=" * 60)
    logger.info("Starting SFT training...")
    logger.info(f"  Data: {data_path.name}")
    logger.info(f"  Training samples: {len(train_dataset):,}")
    logger.info(f"  Eval samples: {len(eval_dataset):,}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: 0.01")
    logger.info(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"  Max seq length: {args.max_seq_length}")
    logger.info(f"  Save every: {save_steps} optimizer steps")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    # ── Resume from checkpoint if requested ────────────────────────
    import time
    resume_ckpt = None
    if args.resume:
        # Find latest checkpoint in output_dir
        ckpt_dirs = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if ckpt_dirs:
            resume_ckpt = str(ckpt_dirs[-1])
            logger.info(f"Resuming from checkpoint: {resume_ckpt}")
        else:
            logger.warning("--resume specified but no checkpoints found. Starting fresh.")

    train_start = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    train_elapsed = time.time() - train_start
    logger.info(f"Training completed in {train_elapsed/3600:.1f} hours ({train_elapsed:.0f}s)")

    # ── Save final model ──────────────────────────────────────────────
    logger.info(f"Saving final LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    text_tokenizer.save_pretrained(output_dir)

    # ── Clean up intermediate checkpoints to save disk space ──────────
    import shutil
    for item in Path(output_dir).iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            logger.info(f"Cleaning up intermediate checkpoint: {item.name}")
            try:
                shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {item.name}: {e}")

    # Save training metadata
    # Find final train and eval loss from log history
    train_loss = None
    eval_loss = None
    for entry in reversed(trainer.state.log_history):
        if train_loss is None and "train_loss" in entry:
            train_loss = entry["train_loss"]
        if eval_loss is None and "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
        if train_loss is not None and eval_loss is not None:
            break

    meta = {
        "model_name": MODEL_NAME,
        "data_file": str(data_path),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "test_samples_excluded": 1500,
        "save_steps": save_steps,
        "final_train_loss": train_loss,
        "final_eval_loss": eval_loss,
        "total_steps": trainer.state.global_step,
        "training_time_hours": round(train_elapsed / 3600, 2),
    }
    meta_path = Path(output_dir) / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Training metadata saved to: {meta_path}")

    logger.info("✓ SFT training complete!")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for Qwen3.5-4B")
    parser.add_argument("--data", type=str, default=None, help="Path to SFT JSONL data")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (default: 1, lower=less OOM risk)")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N optimizer steps (default: 50)")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length (default: 4096)")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (default: 64)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for LoRA adapter")
    parser.add_argument("--dry-run", action="store_true", help="Check data only (no model)")
    parser.add_argument("--dry-run-format", action="store_true", help="Load model + format data, no training")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
