#!/usr/bin/env python3
"""DPO-Uniform training for clinical hallucination reduction (Phase 4D).

Trains Qwen3.5-4B (base or SFT) with QLoRA + DPO on expert-annotated golden
preference pairs. Part of the 12-run ablation study (2 base models × 2 methods
× 3 data sizes).

Usage:
    conda activate vinhthesis

    # Dry run (data only, no model load):
    python scripts/train_dpo.py --base-model qwen35_4b --n-pairs 10 --dry-run

    # Full run — base model, 100 pairs:
    python scripts/train_dpo.py --base-model qwen35_4b --n-pairs 100 \\
        --output-dir models/qwen35_4b_base_dpo_100_lora/

    # Full run — SFT model, 50 pairs:
    python scripts/train_dpo.py --base-model qwen35_4b_sft --n-pairs 50 \\
        --output-dir models/qwen35_4b_sft_dpo_50_lora/

Output:
    models/<name>_lora/          ← LoRA adapter weights
    models/<name>_lora/training.log
    models/<name>_lora/training_metrics.csv
    models/<name>_lora/training_meta.json
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

# ── VRAM fragmentation fix (must be set BEFORE torch import) ──────────
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

for _noisy in ("httpx", "urllib3", "huggingface_hub", "httpcore", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── Constants ─────────────────────────────────────────────────────────

# HuggingFace model ID for base Qwen3.5-4B
BASE_MODEL_HF = "Qwen/Qwen3.5-4B"

# Local merged SFT model (bf16 full weights, created by merge_lora.py)
SFT_MODEL_PATH = PROJECT_ROOT / "models" / "qwen35_4b_sft_merged"

GOLDEN_DIR = PROJECT_ROOT / "data" / "processed" / "dpo" / "golden"

VALID_SIZES = [10, 50, 100]
VALID_BASE_MODELS = ["qwen35_4b", "qwen35_4b_sft"]

SYSTEM_PROMPT = (
    "You are an expert medical professional specializing in clinical documentation. "
    "Summarize the following clinical notes into a concise Brief Hospital Course. "
    "Focus only on clinically significant events, diagnoses, and treatments. "
    "Ensure the summary is technically accurate and uses professional medical terminology. "
    "Strictly adhere to the provided context. If a piece of information is not explicitly "
    "stated in the input text, do not include it. Avoid inferring results or patient outcomes."
)


# ── Data Loading ──────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dpo_pairs(n_pairs: int) -> list[dict]:
    """Load the nested golden DPO pairs for a given subset size."""
    path = GOLDEN_DIR / f"preference_pairs_{n_pairs}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"DPO pairs not found: {path}\n"
            "Run: python scripts/build_golden_dpo_dataset.py"
        )
    pairs = load_jsonl(path)
    logger.info(f"Loaded {len(pairs)} DPO pairs from {path.name}")

    # Quick sanity check
    for i, p in enumerate(pairs):
        if not p.get("prompt") or not p.get("chosen") or not p.get("rejected"):
            raise ValueError(f"Pair {i} is missing prompt/chosen/rejected")
    return pairs


def load_validation_pairs() -> list[dict]:
    path = GOLDEN_DIR / "validation_pairs.jsonl"
    if not path.exists():
        logger.warning("Validation pairs not found — skipping eval split")
        return []
    pairs = load_jsonl(path)
    logger.info(f"Loaded {len(pairs)} validation pairs")
    return pairs


def format_dataset(pairs: list[dict], tokenizer) -> "Dataset":
    """Convert raw pairs to HuggingFace Dataset with chat-formatted prompts.

    TRL DPOTrainer expects columns: prompt, chosen, rejected.
    Prompt is already formatted in the JSONL (instruction + clinical context).
    Chosen/rejected are the summary texts.

    We apply the chat template to prompt+chosen and prompt+rejected separately
    so DPOTrainer sees complete formatted conversations.
    """
    from datasets import Dataset  # noqa: PLC0415

    formatted = []
    for pair in pairs:
        # Build full conversation for chosen
        messages_chosen = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["chosen"]},
        ]
        # Build full conversation for rejected
        messages_rejected = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["rejected"]},
        ]
        # Prompt-only for the "prompt" column (no assistant turn)
        messages_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["prompt"]},
        ]

        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        chosen_text = text_tokenizer.apply_chat_template(
            messages_chosen, tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
        rejected_text = text_tokenizer.apply_chat_template(
            messages_rejected, tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
        prompt_text = text_tokenizer.apply_chat_template(
            messages_prompt, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

        formatted.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })

    logger.info(f"Formatted {len(formatted)} DPO pairs")
    return Dataset.from_list(formatted)


# ── Training ──────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Run DPO-Uniform training."""

    # ── Resolve output dir ────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = f"qwen35_4b_{args.base_model.replace('qwen35_4b_', '')}_dpo_{args.n_pairs}_lora"
        if args.base_model == "qwen35_4b":
            suffix = f"qwen35_4b_base_dpo_{args.n_pairs}_lora"
        output_dir = PROJECT_ROOT / "models" / suffix
    logger.info(f"Output dir: {output_dir}")

    # ── Resolve model path ────────────────────────────────────────────
    if args.base_model == "qwen35_4b":
        model_name_or_path = BASE_MODEL_HF
        logger.info(f"Base model: {model_name_or_path} (HuggingFace)")
    else:
        model_name_or_path = str(SFT_MODEL_PATH)
        if not SFT_MODEL_PATH.exists():
            logger.error(
                f"SFT merged model not found: {SFT_MODEL_PATH}\n"
                "Run: conda activate vinhthesis && python scripts/merge_lora.py"
            )
            sys.exit(1)
        logger.info(f"SFT model: {model_name_or_path}")

    # ── Load data (fail fast before heavy model load) ─────────────────
    train_pairs = load_dpo_pairs(args.n_pairs)
    val_pairs = load_validation_pairs()

    if args.dry_run:
        logger.info(f"[DRY RUN] {len(train_pairs)} train pairs, {len(val_pairs)} val pairs")
        logger.info(f"[DRY RUN] base_model={args.base_model}, n_pairs={args.n_pairs}")
        logger.info(f"[DRY RUN] output_dir={output_dir}")
        logger.info("[DRY RUN] No model loaded. All good!")
        # Print a sample pair
        p = train_pairs[0]
        logger.info(f"[DRY RUN] Sample prompt (first 200c): {p['prompt'][:200]}")
        logger.info(f"[DRY RUN] Sample chosen (first 100c): {p['chosen'][:100]}")
        logger.info(f"[DRY RUN] Sample rejected (first 100c): {p['rejected'][:100]}")
        return

    # ── Heavy imports (AFTER dry-run exit, Unsloth MUST come first) ───
    from unsloth import FastLanguageModel  # noqa: PLC0415
    import torch  # noqa: PLC0415

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

    from datasets import Dataset  # noqa: PLC0415, F401
    from trl import DPOTrainer, DPOConfig  # noqa: PLC0415
    from transformers import TrainerCallback  # noqa: PLC0415

    class CSVLoggingCallback(TrainerCallback):
        """Write step-level metrics to CSV. Flushed per-step for crash safety."""

        def __init__(self, csv_path: Path) -> None:
            self.csv_path = csv_path
            self._file = None
            self._writer = None

        def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow([
                "step", "epoch", "train_loss", "eval_loss",
                "reward_accuracy", "learning_rate", "timestamp",
            ])
            self._file.flush()
            logger.info(f"CSV log → {self.csv_path}")

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if logs is None or self._writer is None:
                return
            row = [
                state.global_step,
                f"{state.epoch:.3f}" if state.epoch else "",
                logs.get("loss", logs.get("train_loss", "")),
                logs.get("eval_loss", ""),
                logs.get("rewards/accuracies", logs.get("reward_accuracy", "")),
                logs.get("learning_rate", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ]
            self._writer.writerow(row)
            if self._file:
                self._file.flush()

        def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
            if self._file:
                self._file.close()
                logger.info(f"CSV log closed: {self.csv_path}")

    # ── Load model ────────────────────────────────────────────────────
    logger.info(f"Loading model from: {model_name_or_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total_params:,} "
        f"({100 * trainable / total_params:.2f}%)"
    )

    # ── VRAM check ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc; gc.collect()  # noqa: PLC0415, E702
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        logger.info(
            f"VRAM: {vram_used:.1f}GB used / {vram_total:.1f}GB total "
            f"({vram_total - vram_used:.1f}GB free)"
        )

    # ── Format dataset ────────────────────────────────────────────────
    train_dataset = format_dataset(train_pairs, tokenizer)
    eval_dataset = format_dataset(val_pairs, tokenizer) if val_pairs else None

    # ── Setup output dir + logging ────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "training.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"File logging → {output_dir / 'training.log'}")

    csv_callback = CSVLoggingCallback(csv_path=output_dir / "training_metrics.csv")

    # ── Get underlying text tokenizer ─────────────────────────────────
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # ── DPO Config ────────────────────────────────────────────────────
    # Compute total steps for save_steps calculation
    total_samples = len(train_dataset)
    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, total_samples // eff_batch)
    total_steps = steps_per_epoch * args.epochs
    # Save every N steps (default: every 2 steps, or every step if very few)
    computed_save_steps = min(args.save_steps, max(1, total_steps // 2))
    logger.info(
        f"Steps: {steps_per_epoch}/epoch × {args.epochs} epochs = {total_steps} total, "
        f"saving every {computed_save_steps} steps"
    )

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        optim="adamw_8bit",          # 8-bit Adam → halves optimizer state VRAM
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,           # log every step (tiny dataset)
        save_strategy="steps",
        save_steps=computed_save_steps,
        eval_strategy="epoch" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=False,  # we save final adapter manually
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        precompute_ref_log_probs=True,  # pre-compute ref logps → halves peak VRAM
        seed=42,
        report_to="none",
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
    )

    # ── DPO Trainer ───────────────────────────────────────────────────
    # FIX: Unsloth's patched DPOTrainer checks model.config.model_type against
    # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES. Qwen3.5-4B is a text-only model
    # but its model_type matches the vision mapping → KeyError: 'images'.
    # Workaround: temporarily remove model_type from the vision mapping.
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES  # noqa: PLC0415
    _model_type = model.config.model_type
    _removed = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.pop(_model_type, None)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=text_tokenizer,  # must be raw tokenizer, NOT the VLM processor
        callbacks=[csv_callback],
    )

    # Restore the mapping (clean up monkey-patch)
    if _removed is not None:
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES[_model_type] = _removed

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting DPO-Uniform training")
    logger.info(f"  Base model  : {args.base_model} ({model_name_or_path})")
    logger.info(f"  Train pairs : {len(train_dataset)}")
    logger.info(f"  Val pairs   : {len(eval_dataset) if eval_dataset else 0}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  Batch       : {args.batch_size} × {args.grad_accum} = "
                f"{args.batch_size * args.grad_accum} effective")
    logger.info(f"  LR          : {args.lr}  warmup={args.warmup_ratio}")
    logger.info(f"  Beta (DPO)  : {args.beta}  label_smoothing={args.label_smoothing}")
    logger.info(f"  LoRA r/α    : {args.lora_r}/{args.lora_alpha}")
    logger.info(f"  Save every  : {computed_save_steps} steps (total {total_steps})")
    logger.info(f"  Output      : {output_dir}")
    if args.resume:
        logger.info("  Resume      : YES — loading from latest checkpoint")
    logger.info("=" * 60)

    train_start = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    train_elapsed = time.time() - train_start
    logger.info(f"Training done in {train_elapsed / 60:.1f} min ({train_elapsed:.0f}s)")

    # ── Overfitting watchdog ──────────────────────────────────────────
    # Extract final reward_accuracy from log history
    final_reward_acc = None
    for entry in reversed(trainer.state.log_history):
        if "rewards/accuracies" in entry:
            final_reward_acc = entry["rewards/accuracies"]
            break
        if "reward_accuracy" in entry:
            final_reward_acc = entry["reward_accuracy"]
            break

    if final_reward_acc is not None:
        logger.info(f"Final reward_accuracy: {final_reward_acc:.4f}")
        if final_reward_acc > 0.90:
            logger.warning(
                f"⚠ reward_accuracy={final_reward_acc:.3f} > 0.90 — possible overfitting! "
                "Consider reducing epochs or increasing beta."
            )
        elif final_reward_acc < 0.55:
            logger.warning(
                f"⚠ reward_accuracy={final_reward_acc:.3f} < 0.55 — DPO may not be learning. "
                "Consider reducing beta or checking data quality."
            )
        else:
            logger.info("✓ reward_accuracy in healthy range [0.55, 0.90]")

    # ── Save adapter ──────────────────────────────────────────────────
    logger.info(f"Saving LoRA adapter → {output_dir}")
    model.save_pretrained(str(output_dir))
    text_tokenizer.save_pretrained(str(output_dir))

    # ── Clean up intermediate checkpoints to save disk space ──────────
    import shutil
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            logger.info(f"Cleaning up intermediate checkpoint: {item.name}")
            try:
                shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {item.name}: {e}")

    # ── Save training metadata ────────────────────────────────────────
    final_train_loss = None
    final_eval_loss = None
    for entry in reversed(trainer.state.log_history):
        if final_train_loss is None and "train_loss" in entry:
            final_train_loss = entry["train_loss"]
        if final_eval_loss is None and "eval_loss" in entry:
            final_eval_loss = entry["eval_loss"]
        if final_train_loss is not None and final_eval_loss is not None:
            break

    meta = {
        "script": "train_dpo.py",
        "base_model": args.base_model,
        "model_path": model_name_or_path,
        "n_pairs": args.n_pairs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        "max_grad_norm": 1.0,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
        "save_steps": computed_save_steps,
        "train_pairs": len(train_dataset),
        "val_pairs": len(eval_dataset) if eval_dataset else 0,
        "total_steps": trainer.state.global_step,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "final_reward_accuracy": final_reward_acc,
        "training_time_minutes": round(train_elapsed / 60, 2),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved → {meta_path}")
    logger.info("✓ DPO-Uniform training complete!")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DPO-Uniform training for clinical hallucination reduction. "
            "Part of 12-run ablation study (Phase 4D)."
        )
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        choices=VALID_BASE_MODELS,
        help="Base model: 'qwen35_4b' (HuggingFace) or 'qwen35_4b_sft' (local merged)",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        required=True,
        choices=VALID_SIZES,
        help="Number of golden DPO training pairs (10, 50, or 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for LoRA adapter (auto-named if not set)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (default: 1 — DPO overfits fast on small data)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature beta (default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size (default: 1)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16, effective batch=16)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1280,
        help="Max sequence length (default: 1280, covers all golden pairs)",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1100,
        help="Max prompt length for DPO (default: 1100, leaves 180 for response)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank (default: 32, same as SFT)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha (default: 64 = 2× rank)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data + config only, no model load",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio — fraction of total steps for LR warmup (default: 0.1)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="DPO label smoothing (default: 0.0; try 0.1 for noisy data)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=2,
        help="Save checkpoint every N optimizer steps (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
