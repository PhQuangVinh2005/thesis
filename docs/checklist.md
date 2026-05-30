# Thesis Progress — Master Checklist

> Last updated: 2026-05-30
> Novel contribution: **Severity-Weighted DPO (SW-DPO)** — per-sample margins from expert hallucination categories
> Use this file to resume work in a new session. Run: "Use concise-planning to read and understand codebase."
> **Current status**: DPO/SW-DPO scripts written and debugged. Blackwell env rebuilt (`vinhthesis2`). **Next task: re-run DPO smoke test (OOM fix applied), then 12-variant ablation (Phase 4D Step 4).**

---

## Phase 1: Baseline Experiments ✅ COMPLETE

- [x] Preprocess MIMIC-IV-BHC dataset (3 token ranges: 0-1K, 1K-2K, 2K-4K × 500 samples each)
- [x] Configure 5 models (BioMistral-7B, BioMistral-7B-SLERP, Qwen3.5-2B/4B/9B)
- [x] Run baseline inference (5 models × 3 ranges = 15 runs, **500 samples/run**)
- [x] Run few-shot inference (1/5/10-shot × 5 models × 3 ranges = 45 runs, **500 samples/run**)
- [x] Run faithfulness evaluation — baseline (SummaC + AlignScore, 15 runs, **100 samples/run**)

## Phase 2: CoVe Experiments ✅ COMPLETE

- [x] Implement CoVe pipeline (`src/techniques/cove.py`)
- [x] Design CoVe prompts (plan + verify+refine)
- [x] Determine BioMistral incompatibility with CoVe (documented in `docs/known-issues.md`)
- [x] Run CoVe experiments (3 Qwen models × 3 ranges = 9 runs, **500 samples/run**)
- [x] Run faithfulness evaluation — CoVe (**100 samples/run**)

## Phase 3: Remaining Evaluations ✅ COMPLETE

- [x] Run faithfulness evaluation — few-shot (45 runs, **100 samples/run**)
- [x] Run completeness evaluation — ALL experiments (ROUGE/BLEU/BERTScore/MEDCON, **100 samples/run**)
- [x] Run completeness evaluation — CoVe (**100 samples/run**)

> **NOTE**: Phase 3 evaluations ran on first 100 of 500 samples per run. Full 500-sample faithfulness
> evaluation was completed separately for DPO dataset construction (see Phase 4A).

## Phase 4: DPO Finetuning 🔄 IN PROGRESS

> Full plan: `docs/dpo-implementation.md`
> Methods decision log: `docs/hallucination-methods.md`
> Composite score analysis: `docs/composite-score-analysis.md`

### Phase 4A: Auto-Generated DPO Dataset ✅ COMPLETE (archived)

- [x] Run full 500-sample faithfulness eval for baseline (scores → `data/raw/finetune/dpo/baseline/`)
- [x] Write `scripts/build_dpo_dataset.py`
  - [x] Merge `predictions.jsonl` + `faith_scores.jsonl` across 5 models per sample_id
  - [x] Compute composite faithfulness score: `0.6 × alignscore + 0.4 × summac_conv`
  - [x] Select chosen (highest) and rejected (lowest) per sample_id
  - [x] Filter pairs with score gap < 0.05
  - [x] Filter degenerate copy-paste outputs (summary/context ratio > 0.5)
  - [x] Output: `data/processed/dpo/preference_pairs.jsonl`
- [x] Run script and validate output → **1,464 pairs** (from 1,500 candidates)
- [x] Manually inspect 10-20 preference pairs for sanity check
- [x] Generate dataset statistics → `data/processed/dpo/dataset_stats.json`

> **Archived**: Superseded by golden expert pairs (Phase 4A2). Kept for potential "auto vs expert" comparison.

### Phase 4A2: Golden Expert DPO Dataset ✅ COMPLETE

- [x] Write `scripts/build_golden_dpo_dataset.py`
  - [x] Source: Hegselmann et al. (CHIL 2024) — 100 doctor-annotated BHC→patient summary pairs
  - [x] Chosen: `cleaned_improved` summaries (hallucinations removed by doctors)
  - [x] Rejected: `original` summaries (with hallucinations)
  - [x] Nested subsets: 10 ⊂ 50 ⊂ 100 (seed=42) for ablation
  - [x] 10 held-out validation pairs
- [x] Output: `data/processed/dpo/golden/preference_pairs_{10,50,100}.jsonl`
- [x] Validation: `data/processed/dpo/golden/validation_pairs.jsonl`

> **Golden Dataset**: 100 expert pairs. Quality > quantity. Task mismatch (BHC→patient summary vs
> notes→BHC) accepted — faithfulness signal transfers across clinical NLP tasks.

### Phase 4B: Migrate Qwen3.5-4B to HuggingFace ✅ DONE

- [x] Write `UnslothModel` backend (`src/models/unsloth_model.py`)
- [x] Register `"unsloth"` backend in `ModelFactory`
- [x] Create model config: `configs/models/qwen3_5_4b_unsloth.yaml`
- [x] Create experiment config: `configs/experiment/qwen3_5_4b_unsloth.yaml`
- [x] Create requirements: `requirements/finetune.txt`
- [x] Write validation script: `scripts/validate_unsloth.py`
- [x] Update `docs/setup.md` with Unsloth setup instructions
- [x] Install Unsloth + dependencies
- [x] Run 5-sample validation — quality comparable to Ollama (2.03x length ratio = style difference, not quality issue)

### Phase 4C: SFT Training (Stage 1) ✅ COMPLETE

> **Script**: `scripts/train_sft.py`
> **Data**: `data/processed/sft/train_30k.jsonl` (extracted via `scripts/extract_sft_data.py`)
> **Output**: `models/qwen35_4b_sft_lora/` (163MB LoRA adapter + training logs)

- [x] Write `scripts/extract_sft_data.py` with quality checks
  - [x] Excludes 1,500 test set IDs
  - [x] Filters target tokens (50-2000 range)
  - [x] Caps total tokens ≤ 3800 (fits 4096 max_seq_length)
  - [x] Post-write validation (JSON integrity + required fields)
  - [x] tqdm progress + extraction report
- [x] Extract 30K SFT samples → `data/processed/sft/train_30k.jsonl`
- [x] Write `scripts/train_sft.py` with production safeguards
  - [x] Unsloth QLoRA: r=32, alpha=64, all linear layers
  - [x] Chat-template formatting (system + user + assistant)
  - [x] Crash recovery: `save_strategy="steps"`, `--resume` flag, OOM-safe eval batch
  - [x] Persistent logging: `training.log` + `training_metrics.csv` + `training_meta.json`
  - [x] VRAM safety: `PYTORCH_CUDA_ALLOC_CONF`, `empty_cache()`, `per_device_eval_batch_size=1`
  - [x] `load_best_model_at_end=True` (eval_loss)
  - [x] `--dry-run` (data only) and `--dry-run-format` (data + model) modes
- [x] Dry-run validation passed (0 samples exceeding max_seq_length)
- [x] **SFT training complete** (94.86 hours, May 7–10)
  - Config: 28,500 train / 1,500 eval, batch=1×16=16 effective, lr=2e-4, cosine scheduler
  - Total: 5,346 optimizer steps (3 epochs), checkpoints every 50 steps
  - **Final train loss: 0.973** | **Final eval loss: 0.942** (no overfitting)
  - Loss trajectory: 3.02 → 0.97 (train), eval stable at ~0.94
  - Output: `adapter_model.safetensors` (163MB), 5 checkpoints retained
- [x] **LoRA merged + vLLM inference complete** (Blackwell fix: `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_DISABLE_FLASHINFER=1`, `attn_implementation=eager`)
  - `outputs/baseline/qwen3_5_4b_sft/range_{0_1k,1k_2k,2k_4k}/predictions.jsonl` (500 samples each)
  - Added `repetition_penalty=1.2` (36% of samples had severe repetition without it)
- [ ] **🔄 Evaluate SFT model on test set (in progress)**
  ```bash
  # Run in order (separate envs):
  conda activate eval_summac && python scripts/run_evaluation.py \
      --experiment-dir outputs/baseline/qwen3_5_4b_sft/ --phase faithfulness --metrics summac
  conda activate eval_align && python scripts/run_evaluation.py \
      --experiment-dir outputs/baseline/qwen3_5_4b_sft/ --phase faithfulness --metrics alignscore
  conda activate vinhthesis && python scripts/run_evaluation.py \
      --experiment-dir outputs/baseline/qwen3_5_4b_sft/ --phase completeness
  ```

### Phase 4D: DPO Ablation Study — 12 Training Runs

> **12 runs**: 2 base models × 2 methods × 3 data sizes
> Base models: Qwen3.5-4B (base HF) and Qwen3.5-4B SFT (from `models/qwen35_4b_sft_merged/`)
> Methods: DPO-Uniform and SW-DPO-Severity
> Data sizes: 10, 50, 100 golden pairs (nested subsets, seed=42)
> Full design: `docs/dpo-implementation.md`

#### Step 1: Add severity scores to golden dataset

- [x] Write `scripts/add_severity_to_dpo_dataset.py`
  - [x] Load `hallucinations_mimic_di.jsonl` → build lookup: `text → list of label spans`
  - [x] For each of 100 pairs: match by `text` field, count spans per category
  - [x] Compute `severity_score = Σ SEVERITY_WEIGHTS[cat] × count(cat)` (0.0 if no match)
  - [x] Normalize: `norm_severity = severity_score / max_severity_in_dataset` → [0, 1]
  - [x] Add fields: `severity_score`, `norm_severity`, `category_counts`, `severity_margin` (= α × norm_severity with α=1.0)
  - [x] Output: `data/processed/dpo/golden/preference_pairs_100_severity.jsonl` ✅
  - [x] Verified: **100/100 matched**, min=0.00, max=26.50, mean=7.03, std=6.01
        Distribution: 48 low (0-0.2), 29 moderate (0.2-0.4), 12 mid (0.4-0.6), 7 high (0.6-0.8), 4 critical (0.8+)

#### Step 2: Write `scripts/train_dpo.py` (DPO-Uniform) ✅

- [x] CLI flags: `--base-model {qwen35_4b,qwen35_4b_sft}`, `--n-pairs {10,50,100}`, `--output-dir`
- [x] Load model via Unsloth `FastLanguageModel.from_pretrained()`
  - [x] base: `"Qwen/Qwen3.5-4B"`, sft: `models/qwen35_4b_sft_merged/`
- [x] Attach QLoRA adapters: r=32, alpha=64, all linear layers
- [x] Load golden pairs from `data/processed/dpo/golden/preference_pairs_{n}.jsonl`
- [x] Format dataset with chat template: `{prompt, chosen, rejected}` → HuggingFace Dataset
- [x] TRL `DPOConfig`: beta=0.1, lr=5e-6, 1 epoch, batch=1, grad_accum=16, bf16=True
- [x] Overfitting watchdog: logs warning if `reward_accuracy > 0.90` or `< 0.55`
- [x] Save adapter + `training_meta.json` + `training.log` + `training_metrics.csv`
- [x] `--dry-run` mode validated: both `qwen35_4b` and `qwen35_4b_sft` variants ✅
- [x] Lint: Ruff ✅ Bandit ✅ MyPy ✅
- [x] **Blackwell debugging** (May 16–18):
  - [x] Fix Unsloth VLM detection: pop model_type from `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`
  - [x] Fix VLM processor routing: pass `text_tokenizer` not processor wrapper
  - [x] Fix OOM: `precompute_ref_log_probs=True`, `max_seq_length` 4096→2048
  - [x] Rebuild env → `vinhthesis2` (CUDA 12.8, native cuda-toolkit)
  - [x] Fix cuBLAS `cublasCreate` ALLOC_FAILED: pre-initialize/warm up handle early + clean `PYTORCH_CUDA_ALLOC_CONF`
- [x] **Smoke test** (10 pairs) passed! ✅ (Completed in 88s with successful evaluation)

#### Step 3: Write `scripts/train_swdpo.py` (SW-DPO) ✅

- [x] Subclass TRL `DPOTrainer` → `_SWDPOTrainer`
  - [x] Override `get_batch_loss_metrics()`: `logits = β·(chosen_logr - rejected_logr) - α·severity_margin`
  - [x] Loss: `-logsigmoid(logits).mean()` — standard DPO form with shifted margin
- [x] Load severity dataset from `preference_pairs_100_severity.jsonl`; slice first n for subsets
- [x] CLI flags same as `train_dpo.py` plus `--alpha` (default 1.0; sweep {0.5, 1.0, 2.0})
- [x] Same QLoRA config, DPOConfig hyperparams; extra CSV column: `severity/mean_margin`
- [x] Dry-run validated: base (n=10, α=1.0) + SFT (n=100, α=2.0) ✅
- [x] Lint: Ruff ✅ Bandit ✅ MyPy ✅
- [x] Same Blackwell fixes as train_dpo.py applied
- [x] **Smoke test** passed! ✅ (SW-DPO completed in 72s with evaluation metrics computed and stored)

#### Step 4: Run all 12 training runs (Scaled to 5 Epochs ⭐)

| Run | Model | Method | Pairs | Output adapter |
|-----|-------|--------|-------|----------------|
| 1 | base | DPO-Uniform | 10 | `models/qwen35_4b_base_dpo_10_lora/` |
| 2 | base | DPO-Uniform | 50 | `models/qwen35_4b_base_dpo_50_lora/` |
| 3 | base | DPO-Uniform | 100 | `models/qwen35_4b_base_dpo_100_lora/` |
| 4 | base | SW-DPO | 10 | `models/qwen35_4b_base_swdpo_10_lora/` |
| 5 | base | SW-DPO | 50 | `models/qwen35_4b_base_swdpo_50_lora/` |
| 6 | base | SW-DPO | 100 | `models/qwen35_4b_base_swdpo_100_lora/` |
| 7 | SFT | DPO-Uniform | 10 | `models/qwen35_4b_sft_dpo_10_lora/` |
| 8 | SFT | DPO-Uniform | 50 | `models/qwen35_4b_sft_dpo_50_lora/` |
| 9 | SFT | DPO-Uniform | 100 | `models/qwen35_4b_sft_dpo_100_lora/` |
| 10 | SFT | SW-DPO | 10 | `models/qwen35_4b_sft_swdpo_10_lora/` |
| 11 | SFT | SW-DPO | 50 | `models/qwen35_4b_sft_swdpo_50_lora/` |
| 12 | SFT | SW-DPO | 100 | `models/qwen35_4b_sft_swdpo_100_lora/` |

- [x] Configure training scripts to run **5 epochs** instead of 1 (guarantees healthy preference learning gradient budget of 5 to 35 steps under batch size 16)
- [x] Implement post-training recursive directory sweep to automatically delete intermediate checkpoints (safely reclaiming **8.9 GB** of workspace storage)
- [ ] Run 1-6 (base model): ~3 hrs total training (5 epochs)
- [ ] Run 7-12 (SFT model): ~3 hrs total training (5 epochs)
- [ ] Monitor reward_accuracy progression (expect 0.70+ convergence at step 10+)

#### Step 5: Merge + Inference + Evaluation (per variant)

- [ ] For each of 12 adapters: `python scripts/merge_lora.py --adapter models/<name>_lora --output models/<name>_merged`
- [ ] For each of 12 merged models: run vLLM inference (3 ranges × 500 = 1,500 samples)
  - `VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_FLASHINFER=1 python scripts/run_sft_inference.py --model-dir models/<name>_merged --output-dir outputs/dpo/<name>/`
- [ ] For each of 12 variants: run evaluation (completeness + faithfulness)
- [ ] Sanity check on 10 golden validation samples for each variant
- [ ] **Category-level evaluation**: breakdown errors by hallucination type (NEW metric)

### Phase 4D2: SW-DPO — Core Contribution Summary ⭐

> Novel contribution: Severity-conditioned margins from expert hallucination categories.
> Plan: `docs/dpo-implementation.md` Phase E2
> Severity weights (NCC MERP-derived): contradicted_fact=5.0, medication=4.0, condition=3.5,
> procedure=3.0, number=3.0, time=2.0, location=2.0, name=1.5, word=1.0, other=1.0

- [ ] α hyperparameter sweep: {0.5, 1.0, 2.0} on validation set (if time permits)
- [ ] Category-level analysis: does SW-DPO specifically reduce high-severity errors more than DPO-Uniform?

### Phase 4E: Combined Methods

- [ ] Run best DPO/SW-DPO model through CoVe pipeline (finetuning + inference stacking)
- [ ] Evaluate combined SW-DPO+CoVe (completeness + faithfulness + category breakdown)

## Phase 5: Analysis & Paper ⬜ TODO

- [ ] Build final comparison table (Baseline vs Few-Shot vs CoVe vs SFT vs DPO-Uniform vs SW-DPO vs SW-DPO+CoVe)
- [ ] **Category-level analysis**: breakdown by hallucination type (medication, condition, contradicted_fact)
- [ ] Statistical significance tests across methods
- [ ] Generate visualization plots (bar charts, severity distribution, category heatmaps)
- [ ] Analyze per-range performance (0-1K vs 1K-2K vs 2K-4K)
- [ ] Write results section for thesis
- [ ] Write methodology section for research paper — emphasize SW-DPO novelty

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `CONTEXT.md` | Project overview, current state, next steps — **read first** |
| `docs/dpo-implementation.md` | Full DPO + SW-DPO plan with loss formulations, severity weights, ablation design |
| `docs/hallucination-methods.md` | All 14 methods evaluated with papers, decisions, rationale |
| `docs/composite-score-analysis.md` | Empirical analysis of AlignScore/SummaC weights (60/40 justification) |
| `docs/architecture.md` | Strategy Pattern design, data flow, finetuning pipeline |
| `docs/known-issues.md` | BioMistral+CoVe incompatibility, BERTScore patches |
| `docs/setup.md` | Conda env setup, CUDA, Ollama + Unsloth |
| `scripts/train_sft.py` | SFT training — **read as reference for DPO trainer pattern** |
| `scripts/build_golden_dpo_dataset.py` | DPO pair builder — **extend to add severity scores** |
| `scripts/run_sft_inference.py` | vLLM inference — **reuse for DPO model inference** |
| `scripts/run_evaluation.py` | Evaluation — **reuse for DPO evaluation** |
| `data/processed/dpo/golden/` | Expert DPO pairs (preference_pairs_{10,50,100}.jsonl) |
| `data/raw/medical-expert-annotations-*/hallucination_datasets/hallucinations_mimic_di.jsonl` | Severity label source |
| `models/qwen35_4b_sft_merged/` | SFT merged model (base for SFT-DPO runs) |
| `outputs/baseline/qwen3_5_4b_sft/` | SFT inference predictions ✅ |

## Quick Reference: Key Commands

```bash
# === DPO/SW-DPO Training (use vinhthesis2 env) ===
conda activate vinhthesis2

# Smoke test (10 pairs)
python scripts/train_dpo.py --base-model qwen35_4b --n-pairs 10 \
    --output-dir models/test_dpo_10_lora/

# Full ablation runs
python scripts/train_dpo.py --base-model qwen35_4b --n-pairs 100 \
    --output-dir models/qwen35_4b_base_dpo_100_lora/
python scripts/train_dpo.py --base-model qwen35_4b_sft --n-pairs 100 \
    --output-dir models/qwen35_4b_sft_dpo_100_lora/
python scripts/train_swdpo.py --base-model qwen35_4b_sft --n-pairs 100 --alpha 1.0 \
    --output-dir models/qwen35_4b_sft_swdpo_100_lora/

# === SFT Evaluation ===
conda activate eval_summac && python scripts/run_evaluation.py \
    --experiment-dir outputs/baseline/qwen3_5_4b_sft/ --phase faithfulness --metrics summac
conda activate eval_align && python scripts/run_evaluation.py \
    --experiment-dir outputs/baseline/qwen3_5_4b_sft/ --phase faithfulness --metrics alignscore
conda activate vinhthesis2 && python scripts/run_evaluation.py \
    --experiment-dir outputs/baseline/qwen3_5_4b_sft/ --phase completeness

# === DPO Inference (after training) ===
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_FLASHINFER=1
conda activate vinhthesis2 && python scripts/merge_lora.py \
    --adapter models/<name>_lora --output models/<name>_merged
conda activate vllm && python scripts/run_sft_inference.py \
    --model-dir models/<name>_merged --output-dir outputs/dpo/<name>/

# === Monitor Training ===
tail -f models/<name>_lora/training.log
nvidia-smi
```

## Quick Reference: Key Decisions

| Decision | Reasoning | Doc |
|----------|-----------|-----|
| **DPO/SW-DPO 5-Epoch Scaling** | **Ensures healthy preference learning gradient budget (5–35 steps) under effective batch size 16 for small golden subsets (10-100 pairs)** | `checklist.md` Step 4 |
| **Post-Train Checkpoint Sweep** | **Automatically wipes duplicate checkpoint subdirectories, reclaiming 8.9 GB of storage space** | `checklist.md` Step 4 |
| **vLLM Speculative KV & M-RoPE Patches** | **Unlocks Qwen3.5-4B hybrid Mamba-Attention execution in vLLM V1 on Blackwell** | `known-issues.md` |
| **SW-DPO over standard DPO** | **Novel contribution: severity-conditioned margins from expert annotations** | `hallucination-methods.md` #14 |
| DPO as replication baseline | 40% medical hallucination reduction, fits 16GB | `hallucination-methods.md` #9 |
| Golden expert pairs over auto-generated | Quality > quantity; 100 doctor-annotated hallucinations vs 1,464 metric-based pairs | `checklist.md` Phase 4A2 |
| **SFT before DPO (30K samples)** | Domain adaptation on MIMIC-IV-BHC clinical notes; LoRA literature shows 10-30K is optimal | `checklist.md` Phase 4C |
| max_seq_length=2048 (DPO) | Halves logits VRAM (vocab=152K); clinical summaries fit within 2048 | `train_dpo.py` |
| max_seq_length=4096 (SFT) | 3800 token cap at extraction ensures 100% fit; covers ~96.5% of original dataset | `extract_sft_data.py` |
| `precompute_ref_log_probs` | Pre-computes ref model logps before training; halves peak VRAM during DPO | `train_dpo.py`, `train_swdpo.py` |
| **Fresh env (`vinhthesis2`)** | CUDA 12.8 native toolkit; no symlinks/hacks; bitsandbytes JIT works natively | `docs/setup.md` |
| Severity weights from NCC MERP | No human annotation needed; established patient safety standard | `dpo-implementation.md` Phase E2 |
| 3-way severity ablation | Uniform vs Binary vs Graded — isolates value of severity signal | `dpo-implementation.md` Phase E2 |
| 3-way size ablation (10/50/100 pairs) | Nested subsets isolate effect of data scale; cleaner thesis story | `checklist.md` Phase 4D |
| Task mismatch accepted | Faithfulness signal transfers: same domain, same phenomenon (hallucination) | `checklist.md` Phase 4A2 |
| Composite 60/40 (AlignScore/SummaC) | AlignScore has wider range + higher model spread; 87%+ agreement across weights | `composite-score-analysis.md` |
| Unsloth as training framework | 2× faster, 60% less VRAM, Qwen3.5 support confirmed | `dpo-implementation.md` |
| XFormers (not Flash Attention) | FA2 build fails without CUDA_HOME; XFormers is functionally identical, ~10-15% slower | Runtime decision |
| CoVe Option C | Plan sees draft; verify+refine does NOT see draft | `architecture.md` |
| BioMistral excluded from CoVe | Cannot follow multi-step prompts (5 failure modes documented) | `known-issues.md` |

## Quick Reference: SFT Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen3.5-4B (Unsloth, 4-bit NF4) | Best balance of quality and VRAM |
| LoRA r | 32 | Higher rank for domain adaptation |
| LoRA alpha | 64 | 2× rank (standard ratio) |
| Target modules | All 7 linear layers | Maximum adaptation capacity |
| Training samples | 28,500 (95% of 30K) | Optimal for LoRA SFT |
| Eval samples | 1,500 (5% of 30K) | Early stopping via eval_loss |
| Epochs | 3 | Standard for domain adaptation |
| Effective batch | 16 (1 × 16 grad accum) | Fits 16GB VRAM |
| Learning rate | 2e-4 (cosine + 5% warmup) | Standard for QLoRA |
| Weight decay | 0.01 | Regularization |
| Max seq length | 4096 | Training data filtered to ≤3800 tokens (100% fit) |
| Gradient checkpointing | Unsloth smart offload | Trades compute for VRAM |
| Checkpoints | Every 50 optimizer steps | Crash recovery (OOM resilient) |
| **Final train loss** | **0.973** | From 3.02 at step 10 |
| **Final eval loss** | **0.942** | No overfitting (gap = 0.031) |
| **Training time** | **94.86 hours** | May 7–10, 2026 |
| **Adapter size** | **163MB** | safetensors format |
