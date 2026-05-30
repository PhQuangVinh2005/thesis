#!/usr/bin/env python3
"""SW-DPO training for clinical hallucination reduction (Phase 4D — Core Contribution ⭐).

Severity-Weighted DPO: per-sample margins from expert hallucination annotations.
The DPO loss margin scales with clinical severity of errors in the rejected response.

Loss formula:
    logits = β · (log π_θ(y_w|x)/π_ref - log π_θ(y_l|x)/π_ref) - α · norm_severity
    L = -log σ(logits)

Usage:
    conda activate vinhthesis

    # Dry run:
    python scripts/train_swdpo.py --base-model qwen35_4b --n-pairs 10 --dry-run

    # Full run — SFT model, 100 pairs, default alpha=1.0:
    python scripts/train_swdpo.py --base-model qwen35_4b_sft --n-pairs 100 \\
        --output-dir models/qwen35_4b_sft_swdpo_100_lora/

Output:
    models/<name>_lora/  ← LoRA adapter + training.log + training_metrics.csv + training_meta.json
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

# ── VRAM fragmentation fix (before torch import) ──────────────────────
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

BASE_MODEL_HF = "Qwen/Qwen3.5-4B"
SFT_MODEL_PATH = PROJECT_ROOT / "models" / "qwen35_4b_sft_merged"
GOLDEN_DIR = PROJECT_ROOT / "data" / "processed" / "dpo" / "golden"
SEVERITY_DATASET = GOLDEN_DIR / "preference_pairs_100_severity.jsonl"

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


def load_severity_pairs(n_pairs: int) -> list[dict]:
    """Load severity-annotated pairs from preference_pairs_100_severity.jsonl.

    Always loads from the 100-pair severity file and takes the first n_pairs,
    since nesting 10⊂50⊂100 is preserved by seed=42 in build_golden_dpo_dataset.py.
    """
    if not SEVERITY_DATASET.exists():
        raise FileNotFoundError(
            f"Severity dataset not found: {SEVERITY_DATASET}\n"
            "Run: python scripts/add_severity_to_dpo_dataset.py"
        )
    all_pairs = load_jsonl(SEVERITY_DATASET)
    pairs = all_pairs[:n_pairs]
    logger.info(f"Loaded {len(pairs)} severity pairs (from {SEVERITY_DATASET.name})")

    # Sanity checks
    for i, p in enumerate(pairs):
        if not p.get("prompt") or not p.get("chosen") or not p.get("rejected"):
            raise ValueError(f"Pair {i} missing prompt/chosen/rejected")
        if "severity_margin" not in p:
            raise ValueError(f"Pair {i} missing severity_margin — re-run add_severity_to_dpo_dataset.py")
    return pairs


def load_validation_pairs() -> list[dict]:
    path = GOLDEN_DIR / "validation_pairs.jsonl"
    if not path.exists():
        logger.warning("Validation pairs not found — skipping eval")
        return []
    return load_jsonl(path)


def format_dataset(pairs: list[dict], tokenizer, include_severity: bool = False) -> "Dataset":
    """Format pairs into HuggingFace Dataset for DPO training.

    If include_severity=True, adds 'severity_margin' column so the
    SeverityWeightedDPOTrainer can inject it per-batch.
    """
    from datasets import Dataset  # noqa: PLC0415

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    formatted = []
    for pair in pairs:
        msgs_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["prompt"]},
        ]
        msgs_chosen = msgs_prompt + [{"role": "assistant", "content": pair["chosen"]}]
        msgs_rejected = msgs_prompt + [{"role": "assistant", "content": pair["rejected"]}]

        prompt_text = text_tokenizer.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        chosen_text = text_tokenizer.apply_chat_template(
            msgs_chosen, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        rejected_text = text_tokenizer.apply_chat_template(
            msgs_rejected, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )

        record: dict = {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }
        if include_severity:
            record["severity_margin"] = float(pair.get("severity_margin", 0.0))
        formatted.append(record)

    logger.info(f"Formatted {len(formatted)} pairs (severity={'yes' if include_severity else 'no'})")
    return Dataset.from_list(formatted)


# ── SW-DPO Trainer ────────────────────────────────────────────────────

class SeverityWeightedDPOTrainer:
    """Lazy wrapper — the real class is defined inside train() after heavy imports.

    This placeholder exists only for type-checking purposes.
    The actual SeverityWeightedDPOTrainer subclasses trl.DPOTrainer and is
    constructed inside train() once TRL is imported.
    """


# ── Training ──────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915
    """Run SW-DPO training."""

    # ── Resolve paths ─────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_tag = "base" if args.base_model == "qwen35_4b" else "sft"
        output_dir = PROJECT_ROOT / "models" / f"qwen35_4b_{model_tag}_swdpo_{args.n_pairs}_lora"
    logger.info(f"Output dir: {output_dir}")

    model_name_or_path = (
        BASE_MODEL_HF if args.base_model == "qwen35_4b" else str(SFT_MODEL_PATH)
    )
    if args.base_model == "qwen35_4b_sft" and not SFT_MODEL_PATH.exists():
        logger.error(f"SFT model not found: {SFT_MODEL_PATH}")
        sys.exit(1)
    logger.info(f"Model: {model_name_or_path}")

    # ── Load data (fail fast) ─────────────────────────────────────────
    train_pairs = load_severity_pairs(args.n_pairs)
    val_pairs = load_validation_pairs()
    margins = [p["severity_margin"] for p in train_pairs]
    logger.info(
        f"Severity margins — min={min(margins):.3f} max={max(margins):.3f} "
        f"mean={sum(margins)/len(margins):.3f}  alpha={args.alpha}"
    )

    if args.dry_run:
        logger.info(f"[DRY RUN] {len(train_pairs)} train, {len(val_pairs)} val pairs")
        logger.info(f"[DRY RUN] base_model={args.base_model}, n_pairs={args.n_pairs}, alpha={args.alpha}")
        logger.info(f"[DRY RUN] output_dir={output_dir}")
        logger.info(f"[DRY RUN] Sample severity_margin={train_pairs[0]['severity_margin']:.4f}")
        logger.info("[DRY RUN] No model loaded. All good!")
        return

    # ── Heavy imports (Unsloth MUST come before transformers) ─────────
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

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

    from unsloth import FastLanguageModel  # noqa: PLC0415
    from trl import DPOTrainer, DPOConfig  # noqa: PLC0415
    from transformers import TrainerCallback  # noqa: PLC0415

    # ── Define real SeverityWeightedDPOTrainer ────────────────────────
    class _SWDPOTrainer(DPOTrainer):
        """Injects per-sample severity margin into DPO logits.

        Loss: L = -log σ(β·(chosen_logr - rejected_logr) - α·severity_margin)
        High-severity pairs require a wider decision margin before counting as correct.
        """

        def __init__(self, *a, severity_alpha: float = 1.0, **kw) -> None:
            super().__init__(*a, **kw)
            self.severity_alpha = severity_alpha

        def get_batch_loss_metrics(self, model, batch, train_eval="train"):  # type: ignore[override]
            metrics: dict = {}
            policy_outputs = self.concatenated_forward(model, batch)
            policy_chosen_logps = policy_outputs["chosen_logps"]
            policy_rejected_logps = policy_outputs["rejected_logps"]
            policy_chosen_logits = policy_outputs["mean_chosen_logits"]
            policy_rejected_logits = policy_outputs["mean_rejected_logits"]

            # If precompute_ref_log_probs=True, ref logprobs are stored in batch
            if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                ref_chosen_logps = batch["ref_chosen_logps"]
                ref_rejected_logps = batch["ref_rejected_logps"]
            else:
                with torch.no_grad():
                    if self.ref_model is None:
                        with self.null_ref_context():
                            ref_outputs = self.concatenated_forward(self.model, batch)
                    else:
                        ref_outputs = self.concatenated_forward(self.ref_model, batch)
                    ref_chosen_logps = ref_outputs["chosen_logps"]
                    ref_rejected_logps = ref_outputs["rejected_logps"]

            chosen_r = self.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_r = self.beta * (policy_rejected_logps - ref_rejected_logps)
            sev = batch.get("severity_margin", torch.zeros_like(chosen_r))
            sev = sev.to(chosen_r.device).float()

            logits = chosen_r - rejected_r - self.severity_alpha * sev
            loss = -F.logsigmoid(logits).mean()

            reward_acc = (logits.detach() > 0).float().mean()
            p = "eval_" if train_eval == "eval" else ""
            metrics.update({
                f"{p}rewards/chosen": chosen_r.mean().item(),
                f"{p}rewards/rejected": rejected_r.mean().item(),
                f"{p}rewards/accuracies": reward_acc.item(),
                f"{p}rewards/margins": (chosen_r - rejected_r).mean().item(),
                f"{p}severity/mean_margin": sev.mean().item(),
                f"{p}logits/chosen": policy_chosen_logits.mean().item(),
                f"{p}logits/rejected": policy_rejected_logits.mean().item(),
            })
            return loss, metrics

    # ── CSV Callback ──────────────────────────────────────────────────
    class CSVLoggingCallback(TrainerCallback):
        def __init__(self, csv_path: Path) -> None:
            self.csv_path = csv_path
            self._file = None
            self._writer = None

        def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(
                ["step", "epoch", "train_loss", "eval_loss",
                 "reward_accuracy", "severity_margin", "learning_rate", "timestamp"]
            )
            self._file.flush()

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if logs is None or self._writer is None:
                return
            self._writer.writerow([
                state.global_step,
                f"{state.epoch:.3f}" if state.epoch else "",
                logs.get("loss", logs.get("train_loss", "")),
                logs.get("eval_loss", ""),
                logs.get("rewards/accuracies", ""),
                logs.get("severity/mean_margin", ""),
                logs.get("learning_rate", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])
            if self._file:
                self._file.flush()

        def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
            if self._file:
                self._file.close()

    # ── Load model ────────────────────────────────────────────────────
    logger.info(f"Loading model: {model_name_or_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    if torch.cuda.is_available():
        vt = torch.cuda.get_device_properties(0).total_memory / 1e9
        vu = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"VRAM: {vu:.1f}/{vt:.1f}GB ({vt-vu:.1f}GB free)")

    # ── Format datasets ───────────────────────────────────────────────
    train_dataset = format_dataset(train_pairs, tokenizer, include_severity=True)
    eval_dataset = format_dataset(val_pairs, tokenizer) if val_pairs else None

    # ── Setup logging ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "training.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # ── DPO Config ────────────────────────────────────────────────────
    total_samples = len(train_dataset)
    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, total_samples // eff_batch)
    total_steps = steps_per_epoch * args.epochs
    computed_save_steps = min(args.save_steps, max(1, total_steps // 2))
    logger.info(f"Steps: {steps_per_epoch}/epoch × {args.epochs} = {total_steps}, save every {computed_save_steps}")

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
        logging_steps=1,
        save_strategy="steps",
        save_steps=computed_save_steps,
        eval_strategy="epoch" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=False,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        precompute_ref_log_probs=True,  # pre-compute ref logps → halves peak VRAM
        seed=42,
        report_to="none",
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
    )

    # FIX: Unsloth VLM detection bug (Qwen3.5 matched as vision model)
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES  # noqa: PLC0415
    _model_type = model.config.model_type
    _removed = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.pop(_model_type, None)

    trainer = _SWDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=text_tokenizer,  # raw tokenizer, NOT the VLM processor
        severity_alpha=args.alpha,
        callbacks=[CSVLoggingCallback(output_dir / "training_metrics.csv")],
    )

    if _removed is not None:
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES[_model_type] = _removed

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting SW-DPO training")
    logger.info(f"  base_model={args.base_model}  n_pairs={args.n_pairs}  alpha={args.alpha}  beta={args.beta}")
    logger.info(f"  lr={args.lr}  warmup={args.warmup_ratio}  label_smoothing={args.label_smoothing}")
    logger.info(f"  epochs={args.epochs}  batch={args.batch_size}×{args.grad_accum}  save_steps={computed_save_steps}")
    if args.resume:
        logger.info("  Resume: YES — loading from latest checkpoint")
    logger.info("=" * 60)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed = time.time() - t0
    logger.info(f"Training done in {elapsed/60:.1f} min")

    final_reward_acc = next(
        (e["rewards/accuracies"] for e in reversed(trainer.state.log_history) if "rewards/accuracies" in e),
        None,
    )
    if final_reward_acc is not None:
        logger.info(f"Final reward_accuracy: {final_reward_acc:.4f}")
        if final_reward_acc > 0.90:
            logger.warning("⚠ reward_accuracy > 0.90 — possible overfitting!")
        elif final_reward_acc < 0.55:
            logger.warning("⚠ reward_accuracy < 0.55 — check data quality or lower beta")

    # ── Save ──────────────────────────────────────────────────────────
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

    final_train_loss = next(
        (e["train_loss"] for e in reversed(trainer.state.log_history) if "train_loss" in e),
        None,
    )
    meta = {
        "script": "train_swdpo.py", "method": "SW-DPO",
        "base_model": args.base_model, "model_path": model_name_or_path,
        "n_pairs": args.n_pairs, "alpha": args.alpha, "beta": args.beta,
        "label_smoothing": args.label_smoothing, "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": 1.0,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "epochs": args.epochs, "learning_rate": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length, "max_prompt_length": args.max_prompt_length,
        "save_steps": computed_save_steps,
        "severity_margin_mean": sum(margins) / len(margins),
        "total_steps": trainer.state.global_step,
        "final_train_loss": final_train_loss,
        "final_reward_accuracy": final_reward_acc,
        "training_time_minutes": round(elapsed / 60, 2),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata → {output_dir / 'training_meta.json'}")
    logger.info("✓ SW-DPO training complete!")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SW-DPO: severity-weighted DPO for clinical summarization (Phase 4D ⭐)."
    )
    parser.add_argument("--base-model", type=str, required=True, choices=VALID_BASE_MODELS)
    parser.add_argument("--n-pairs", type=int, required=True, choices=VALID_SIZES)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Severity margin strength α (default 1.0; sweep {0.5,1.0,2.0})")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1280)
    parser.add_argument("--max-prompt-length", type=int, default=1100)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="LR warmup fraction (default: 0.1)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="DPO label smoothing (default: 0.0)")
    parser.add_argument("--save-steps", type=int, default=2,
                        help="Save checkpoint every N steps (default: 2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
