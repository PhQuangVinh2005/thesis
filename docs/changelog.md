# Changelog

## 2026-05-04: Severity-Weighted DPO (SW-DPO) — Methodology Pivot

### Research Problem

Standard DPO (Phase 4D) replicates Hegselmann et al. (CHIL 2024) — using their 100 golden pairs
with uniform margins provides insufficient academic novelty for a thesis contribution. The Hegselmann
dataset includes rich 11-category span-level hallucination annotations that were used only for
analysis in the original paper.

### Novel Methodology: SW-DPO

We propose using expert-annotated hallucination categories as a **training signal** rather than
just an analysis tool. By mapping categories to clinical severity weights derived from the
**NCC MERP patient safety index**, we create per-sample margins in the DPO loss function.

**Key papers cited**:
- F-DPO: Chaduvula et al., arXiv:2601.03027 (factuality-conditioned margins)
- NCC MERP Index for Categorizing Medication Errors, 2001 (severity taxonomy)
- Hegselmann et al., CHIL 2024 (annotation protocol and dataset)

### Data Analysis Results

- 100 doctor-written summaries: **286 total hallucination spans** across **93 samples** (avg 3.1 spans/sample)
- 11 hallucination categories with highly skewed distribution:
  - `word_unsupported` (26.6%) — low clinical risk, weight 1.0
  - `medication_unsupported` (11.9%) + `contradicted_fact` (5.2%) — **highest** clinical risk, weights 4.0/5.0
- LLM-generated summaries: 114 spans across 50/100 samples — different distribution (GPT-4 mostly `word_unsupported`)

### Modified Files

| File | Change |
|------|--------|
| `docs/dpo-implementation.md` | Added Phase E2 (SW-DPO) with loss formulation, severity weights, implementation pseudocode |
| `docs/hallucination-methods.md` | Added Method #14 (SW-DPO), updated thesis contribution statement |
| `docs/checklist.md` | Added Phase 4D2 (SW-DPO ablation), category-level evaluation metric |
| `docs/changelog.md` | This entry |

### Design Decisions

| Decision | Reasoning |
|----------|-----------|
| SW-DPO margin from NCC MERP | Avoids need for human annotation; established patient safety standard |
| 3-way severity ablation (Uniform/Binary/Severity) | Isolates whether graded severity helps vs binary or none |
| Subclass `DPOTrainer` | Minimal code change (~30 lines); only modify `get_batch_loss_metrics()` |
| Keep DPO-Uniform as control | Clean ablation: same data, same hyperparams, only margin differs |

### Key Finding

> The Hegselmann annotations contain a rich, unused training signal. `medication_unsupported` and
> `contradicted_fact` spans are rare (17.1% combined) but clinically dangerous. Standard DPO treats
> these the same as harmless `word_unsupported` (26.6%), wasting optimization budget on low-risk
> errors. SW-DPO corrects this by applying larger margins to high-severity hallucinations.

## 2026-04-29: DPO Finetuning Research & Documentation

### New Files

| File | Purpose |
|------|---------|
| `docs/dpo-implementation.md` | Full implementation plan: SFT→DPO pipeline with code, configs, hardware specs |
| `docs/hallucination-methods.md` | Decision log for 13 hallucination reduction methods with paper citations |
| `docs/checklist.md` | Master progress tracker for all thesis phases |

### Research Decisions

- **DPO** selected as primary finetuning method (Tian et al., ICLR 2024: 40% medical hallucination reduction)
- **Self-generated preference pairs** from existing baseline predictions across 5 models (no public clinical DPO dataset exists)
- **Unsloth + QLoRA** as training framework (confirmed Qwen3.5 support)
- **Two-stage pipeline**: SFT (ground truth) → DPO (LM-generated pairs scored by AlignScore + SummaC)
- **Rejected methods**: Self-Refine (CoVe covers it), CAD (too complex), NER Grounding (entity DB quality), GRPO (unproven for summarization), SPIN (not hallucination-specific)
- **Backup methods**: ORPO (if DPO OOMs), KTO (if preference pairs noisy)

### Key Finding

> No new inference is needed for DPO dataset construction. Existing baseline predictions (5 models × 3 ranges × 500 samples = 7,500 predictions with faithfulness scores) can be reorganized into ~1,500 preference pairs by selecting best/worst faithfulness-scored predictions per sample_id.

## 2026-04-27: Chain-of-Verification (CoVe) Pipeline

### New Files

| File | Purpose |
|------|---------|
| `src/techniques/cove.py` | 3-step CoVe technique: draft → plan → verify+refine |
| `configs/prompts/cove_plan.yaml` | Prompt: generate N verification questions from draft |
| `configs/prompts/cove_verify_refine.yaml` | Prompt: answer questions from source + write corrected summary |
| `configs/experiment/cove/*.yaml` | Experiment configs (5 models) |

### Modified Files

| File | Change |
|------|--------|
| `src/pipelines/summarizer.py` | Persist CoVe intermediates (`cove_draft`, `cove_questions`, `cove_raw_verification`) in JSONL metadata |
| `src/models/hf_model.py` | Graceful fallback: flash_attention_2 → eager, safetensors → pytorch_model.bin |
| `src/techniques/cove.py` | Robust regex extraction for PART 2 summary with 6 marker patterns + verification-only detection |

### Design Decisions

- **Option C**: Plan phase sees draft + context; verify+refine sees context + questions only (no draft leakage)
- **BioMistral excluded from CoVe**: Both BioMistral-7B and BioMistral-7B-SLERP cannot follow the structured multi-step CoVe prompt. They produce degenerate outputs: token dumps as drafts, single trivial questions instead of 5, no PART 1/PART 2 structure, and hallucinated procedures. CoVe runs only on Qwen3.5 (2B/4B/9B).
- **Intermediate persistence**: All CoVe steps stored in JSONL for thesis analysis — does not affect evaluation pipeline.

### Key Finding

> CoVe requires sufficient instruction-following capability. Domain-specific models without instruction tuning (BioMistral) fail at multi-step structured reasoning, while general instruction-tuned models (Qwen3.5) succeed.

## 2026-04-26: CUDA 13.0 Upgrade

- Driver 575 → 580, CUDA 12.9 → 13.0
- PyTorch cu128 → cu130
- Added flash_attn_3 (community wheels cu130/torch2.11.0)
- Removed causal-conv1d (flash-linear-attention provides Triton-based conv1d)
- Enabled flash_attention_2 in TransformersModel

## 2026-03-30: vLLM → Transformers Migration

| Before (vLLM) | After (Transformers) |
|---|---|
| HTTP server + API calls | Model loaded in Python process |
| No quantization (fp16 only) | 8-bit via bitsandbytes |
| Separate server process | Single process |

**Added**: `hf_model.py`, `ollama_model.py`, `test_model.py`
**Removed**: `vllm_model.py`
**Unchanged**: `src/data/`, `src/pipelines/`, `src/techniques/`, `src/prompts/`

## 2026-03-30: Cross-Tokenizer EDA

- BioMistral/Qwen3.5 tokenizers produce ~1.1x tokens vs GPT-4
- GPT-4 token ranges in dataset are valid proxies for all models
- `max_model_len` set to 12288 (covers 10-shot few-shot)
