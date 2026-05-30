# Project Context

> Bachelor thesis: Reducing Hallucinations in Clinical Text Summarization using LLMs.
> **Last updated**: 2026-05-30
> **Current status**: DPO and SW-DPO training scripts successfully written and smoke-tested. Master ablation suite modified to scale DPO preference training to **5 epochs** (enabling active preference updates). Automatic post-training intermediate checkpoint cleanup implemented. **Current task: executing the full 12-variant 5-epoch ablation study (Phase 4D Step 4).**

## Problem

LLMs generate clinically unfaithful content (hallucinations) when summarizing medical records.
This thesis tests hallucination reduction techniques and measures their effectiveness.

## Architecture

**Strategy Pattern OOP** — swap models/techniques/metrics via YAML config.

```
BaseLLM        → TransformersModel (BioMistral)  |  OllamaModel (Qwen3.5)  |  UnslothModel (Qwen3.5-4B finetuning)
BaseTechnique  → BaselineTechnique  |  FewShotTechnique  |  CoVeTechnique
BaseMetric     → ROUGE/BLEU/BERTScore/MEDCON  |  SummaC/AlignScore
```

### CoVe Pipeline (3 LLM calls per sample)

```
Step 1 — DRAFT:          baseline prompt → initial summary (may hallucinate)
Step 2 — PLAN:           draft + context → N verification questions
Step 3 — VERIFY+REFINE:  context + questions only (no draft) → verified summary
```

Design: **Option C** — plan sees draft, verify+refine does NOT see draft (prevents leakage).

### Finetuning Pipeline (SFT → DPO/SW-DPO)

```
Stage 1 — SFT:           30K MIMIC-IV-BHC samples → domain adaptation (QLoRA r=32) ✅ DONE
                         Final: train_loss=0.973, eval_loss=0.942, 94.86 hrs
Stage 2 — DPO-Uniform:   100 expert pairs → preference alignment (3-way size ablation: 10/50/100, 5 epochs)
Stage 3 — SW-DPO:        100 expert pairs + severity weights → novel contribution (5 epochs) ⭐
```

### DPO Ablation Design (12 Runs Total)

```
2 base models × 2 methods × 3 data sizes = 12 training runs (5 epochs each)

Base models:  [Qwen3.5-4B (base)]  ×  [Qwen3.5-4B SFT]
Methods:      [DPO-Uniform]  ×  [SW-DPO-Severity]
Data sizes:   [10 pairs]  ⊂  [50 pairs]  ⊂  [100 pairs]

Per-run est:  ~30 min train + ~5 min merge + ~40 min inference + ~2 hr eval
Total est:    ~38-48 hours (dominated by evaluation)
```

SW-DPO loss: `L = -log σ(β·(logr_chosen - logr_rejected) - α·norm_severity)`
Severity weights: contradicted_fact=5.0, medication=4.0, condition=3.5, procedure=3.0,
                  number=3.0, time=2.0, location=2.0, name=1.5, word=1.0, other=1.0
(α=1.0 by default; sweep {0.5, 1.0, 2.0} on validation set if time permits)

## Environment

| Item | Value |
|------|-------|
| GPU | RTX 5060 Ti 16GB (Blackwell, sm_120) |
| CUDA driver | 13.0 (driver 580.142) / CUDA toolkit 12.4 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.11 (vinhthesis) / 3.12 (vllm env) |
| Conda envs | 4 envs (see below) |

### Conda Environments

| Env | Purpose | Key Packages |
|-----|---------|------|
| `vinhthesis` | Main training + inference | Unsloth, TRL, transformers, PEFT |
| `vllm` | Fast batch inference (Python 3.12) | vLLM, transformers |
| `eval_summac` | SummaC faithfulness metric | summac, spacy |
| `eval_align` | AlignScore faithfulness metric | alignscore |

### Blackwell GPU Workarounds (REQUIRED for vLLM)

Always set before launching vLLM inference:
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
```
Also use `attn_implementation="eager"` in vLLM engine args. Memory: `gpu_memory_utilization=0.80`.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/models/` | LLM backends (factory, base, hf_model, ollama_model, unsloth_model) |
| `src/techniques/` | Hallucination reduction (base, baseline, fewshot, cove) |
| `src/pipelines/summarizer.py` | Summarization orchestration + CoVe metadata persistence |
| `src/evaluation/` | Metrics (completeness + faithfulness) |
| `src/data/schema.py` | Core data types (EvalSample) |
| `scripts/extract_sft_data.py` | Extract 30K SFT samples from MIMIC-IV-BHC (excludes 1,500 test set) |
| `scripts/train_sft.py` | SFT training with QLoRA, crash recovery, CSV logging |
| `scripts/build_golden_dpo_dataset.py` | Build expert DPO pairs (100 nested: 10⊂50⊂100) — needs severity extension |
| `scripts/merge_lora.py` | Merge LoRA adapter → full bf16 model for vLLM (run in vinhthesis) |
| `scripts/run_sft_inference.py` | vLLM batch inference on 1,500 test samples → predictions.jsonl |
| `scripts/run_evaluation.py` | Run completeness (vinhthesis) or faithfulness (eval_summac/eval_align) |
| `scripts/train_dpo.py` | **[NOT YET WRITTEN]** DPO-Uniform training script |
| `scripts/train_swdpo.py` | **[NOT YET WRITTEN]** SW-DPO training script |
| `configs/` | YAML configs (models, experiments, prompts, eval) |
| `docs/dpo-implementation.md` | Full DPO + SW-DPO plan (loss formulas, severity weights, ablation design) |
| `docs/checklist.md` | Master progress checklist — **start here to resume** |
| `data/processed/sft/` | Pre-extracted SFT training data (train_30k.jsonl) |
| `data/processed/dpo/golden/` | Expert DPO preference pairs (preference_pairs_{10,50,100}.jsonl) |
| `models/qwen35_4b_sft_lora/` | SFT LoRA adapter (163MB) + training logs/metrics/meta |
| `models/qwen35_4b_sft_merged/` | Merged bf16 model for vLLM (~8GB, git-ignored) |
| `outputs/baseline/qwen3_5_4b_sft/` | SFT inference predictions (3 ranges × 500 = 1,500 samples) ✅ |
| `data/raw/medical-expert-annotations-*/hallucination_datasets/` | PhysioNet expert annotations (source for SW-DPO severity) |

## Conventions

- `uv pip install` for main env
- Config-driven: swap model/prompt/technique via YAML
- Lazy imports for heavy deps (torch, evaluation metrics, unsloth)
- 4 conda envs due to irreconcilable deps + vLLM isolation
- Unsloth imported BEFORE transformers (required for monkey-patching)
- Finetuning uses pre-extracted JSONL (not raw CSV) for speed + reproducibility
- All training logs persist to disk (training.log + training_metrics.csv) — survives SSH drops
- CoVe only runs on Qwen models (BioMistral lacks multi-step reasoning)
- vLLM inference requires merged model (`merge_lora.py` first, then `run_sft_inference.py`)
- **Blackwell (sm_120)**: always set `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_DISABLE_FLASHINFER=1`
- **Evaluation envs**: SummaC → `eval_summac`, AlignScore → `eval_align`, completeness → `vinhthesis`
- **Repetition penalty**: add `repetition_penalty=1.2` to vLLM SamplingParams (36% of SFT samples had severe repetition without it)

## Current State (2026-05-30)

### ✅ Done
- SFT training complete (Phase 4C): `models/qwen35_4b_sft_lora/` (train_loss=0.973)
- LoRA merged to full bf16: `models/qwen35_4b_sft_merged/` (via `merge_lora.py`)
- **vLLM inference complete**: all 3 ranges (0_1k, 1k_2k, 2k_4k), 1,500 samples total
  - `outputs/baseline/qwen3_5_4b_sft/range_*/predictions.jsonl`
  - Blackwell fix: `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_DISABLE_FLASHINFER=1`, `attn_implementation=eager`
- Expert DPO golden dataset built: `data/processed/dpo/golden/preference_pairs_{10,50,100}.jsonl`
- SW-DPO severity margin dataset built: `preference_pairs_100_severity.jsonl`
- DPO-Uniform (`train_dpo.py`) and SW-DPO-Severity (`train_swdpo.py`) training scripts written, debugged, and fully smoke-tested on Blackwell.
- Patched vLLM V1 spec page-size mismatch and uses_mrope assert to fully unlock hybrid Qwen3.5 execution.
- Added automatic post-training intermediate checkpoint sweeps to reclaim workspace storage.
- Modified the master ablation suite (`run_all_ablation.sh`) to scale preference learning to **5 epochs** across all 12 models to ensure healthy log-probability adaptation.

### 🔄 In Progress
- **Executing the 12-variant 5-epoch DPO/SW-DPO training runs** via `scripts/run_all_ablation.sh`.

### ⬜ Next Steps
- **Step 1**: For each of the 12 resulting LoRA adapters, run `scripts/merge_lora.py` to produce full merged bf16 weights.
- **Step 2**: Run batch vLLM inference across all 3 token ranges (1,500 samples total) for each of the 12 variants.
- **Step 3**: Evaluate all 12 sets of predictions on completeness (vinhthesis2) and faithfulness (eval_summac/eval_align).
- **Step 4**: Perform category-level error breakdown analysis (specifically checking if SW-DPO reduces high-severity medication and factual contradictions more than DPO-Uniform).
