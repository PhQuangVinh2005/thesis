# Thesis Progress — Master Checklist

> Last updated: 2026-05-04 (validated against filesystem)
> Novel contribution: **Severity-Weighted DPO (SW-DPO)** — per-sample margins from expert hallucination categories
> Use this file to resume work in a new session. Run: "Use concise-planning to read and understand codebase."

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

### Phase 4B: Migrate Qwen3.5-4B to HuggingFace

- [ ] Install Unsloth + flash-linear-attention + TRL + PEFT + bitsandbytes
- [ ] Load Qwen3.5-4B via Unsloth (4-bit QLoRA)
- [ ] Run 5-10 sample inference, compare quality with Ollama baseline
- [ ] Document environment setup in `docs/setup.md`

### Phase 4C: SFT Training (Stage 1)

- [ ] Format MIMIC-IV-BHC as instruction-tuning dataset (messages format)
- [ ] Write `scripts/train_sft.py`
- [ ] Run SFT training (r=32, 3 epochs, lr=2e-4, ~3-5 hrs)
- [ ] Save LoRA adapter: `models/qwen35_4b_sft_lora/`
- [ ] Run SFT model inference on test set
- [ ] Evaluate SFT model (completeness + faithfulness)

### Phase 4D: DPO-Uniform Training (Stage 2 — Replication Baseline) — 3-way size ablation

- [ ] Write `scripts/train_dpo.py` (configurable dataset size)
- [ ] Load SFT checkpoint as base for all 3 runs
- [ ] DPO-U run 1: 10 expert pairs → `models/qwen35_4b_dpo_uniform_10_lora/`
- [ ] DPO-U run 2: 50 expert pairs → `models/qwen35_4b_dpo_uniform_50_lora/`
- [ ] DPO-U run 3: 100 expert pairs → `models/qwen35_4b_dpo_uniform_100_lora/`
- [ ] Monitor reward_accuracy per run (target: 0.65-0.85, stop if >0.90)
- [ ] Run all 3 DPO-Uniform models on 1,500 test set (3 ranges × 500)
- [ ] Evaluate all 3 (completeness + faithfulness)
- [ ] Sanity check on 10 golden validation samples

### Phase 4D2: SW-DPO Training (Stage 2 — Core Contribution) — 3-way severity ablation ⭐

> Novel contribution: Severity-conditioned margins from expert hallucination categories.
> Plan: `docs/dpo-implementation.md` Phase E2

- [ ] Update `scripts/build_golden_dpo_dataset.py` to inject severity scores from `hallucinations_mimic_di.jsonl`
  - [ ] Map 11 hallucination categories to NCC MERP severity weights
  - [ ] Compute per-sample `severity_score` = Σ weight(category_i) × count(category_i)
  - [ ] Normalize to [0,1] across dataset → `normalized_severity`
  - [ ] Add `severity_score`, `normalized_severity`, `category_breakdown` to each pair
  - [ ] Output: `data/processed/dpo/golden/preference_pairs_100_severity.jsonl`
- [ ] Write `scripts/train_swdpo.py` (or extend train_dpo.py with --severity flag)
  - [ ] Subclass TRL `DPOTrainer` → `SeverityWeightedDPOTrainer`
  - [ ] Override `get_batch_loss_metrics()` to inject per-sample margin
  - [ ] Support 3 modes: uniform (α=0), binary (α×{0,1}), severity (α×norm_sev)
- [ ] SW-DPO-Binary: 100 pairs, margin={0,1} → `models/qwen35_4b_swdpo_binary_100_lora/`
- [ ] SW-DPO-Severity: 100 pairs, margin=norm_sev → `models/qwen35_4b_swdpo_severity_100_lora/`
- [ ] Sweep α ∈ {0.5, 1.0, 2.0} on validation set to select best margin strength
- [ ] Evaluate all SW-DPO variants (completeness + faithfulness)
- [ ] **Category-level evaluation**: breakdown errors by hallucination type (NEW metric)

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
| `docs/dpo-implementation.md` | Full DPO + SW-DPO plan with loss formulations, severity weights, ablation design |
| `docs/hallucination-methods.md` | All 14 methods evaluated with papers, decisions, rationale |
| `docs/composite-score-analysis.md` | Empirical analysis of AlignScore/SummaC weights (60/40 justification) |
| `docs/architecture.md` | Strategy Pattern design, data flow, model compatibility |
| `docs/known-issues.md` | BioMistral+CoVe incompatibility, BERTScore patches |
| `docs/setup.md` | 3 conda envs, CUDA 13.0, Ollama setup |
| `docs/experiments.md` | CLI reference, config mapping, output structure |
| `docs/evaluation.md` | Two-phase evaluation (completeness + faithfulness) |
| `data/processed/dpo/golden/dataset_stats.json` | Golden DPO dataset statistics (10/50/100 nested subsets) |
| `data/processed/dpo/dataset_stats.json` | Auto-generated DPO dataset statistics (archived, 1,464 pairs) |

## Quick Reference: Key Decisions

| Decision | Reasoning | Doc |
|----------|-----------|-----|
| **SW-DPO over standard DPO** | **Novel contribution: severity-conditioned margins from expert annotations** | `hallucination-methods.md` #14 |
| DPO as replication baseline | 40% medical hallucination reduction, fits 16GB | `hallucination-methods.md` #9 |
| Golden expert pairs over auto-generated | Quality > quantity; 100 doctor-annotated hallucinations vs 1,464 metric-based pairs | `checklist.md` Phase 4A2 |
| Severity weights from NCC MERP | No human annotation needed; established patient safety standard | `dpo-implementation.md` Phase E2 |
| 3-way severity ablation | Uniform vs Binary vs Graded — isolates value of severity signal | `dpo-implementation.md` Phase E2 |
| 3-way size ablation (10/50/100 pairs) | Nested subsets isolate effect of data scale; cleaner thesis story | `checklist.md` Phase 4D |
| Task mismatch accepted | Faithfulness signal transfers: same domain, same phenomenon (hallucination) | `checklist.md` Phase 4A2 |
| Composite 60/40 (AlignScore/SummaC) | AlignScore has wider range + higher model spread; 87%+ agreement across weights | `composite-score-analysis.md` |
| Degenerate output filter (ratio > 0.5) | Prevents copy-paste outputs from being "chosen"; 446 predictions excluded | `build_dpo_dataset.py` |
| Unsloth as training framework | 2× faster, 60% less VRAM, Qwen3.5 support confirmed | `dpo-implementation.md` |
| CoVe Option C | Plan sees draft; verify+refine does NOT see draft | `architecture.md` |
| BioMistral excluded from CoVe | Cannot follow multi-step prompts (5 failure modes documented) | `known-issues.md` |
