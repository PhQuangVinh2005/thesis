# Changelog

## 2026-05-30: Hybrid vLLM Inference & DPO 5-Epoch Scaling (Phase 4D / 4E ⭐)

### Problem

Two critical sets of blockers were addressed to stabilize the production pipeline and ensure academic rigor for the thesis:
1. **vLLM V1 Engine Failures**: Loading the hybrid Mamba-Attention Qwen3.5-4B model on the Blackwell server crashed with memory representation errors and Rotary Embedding asserts.
2. **DPO Preference Learning Duration**: DPO ablation runs completed too fast with default parameters. Small-data training (10, 50, 100 golden pairs) using an effective batch size of 16 completed in only 1–7 optimizer steps, leaving the model's preference alignment at random chance level. At the same time, massive intermediate checkpoints threatened to deplete storage capacity.

### Blockers Resolved & Technical Actions

| # | Error / Finding | Root Cause | Technical Resolution |
|---|-----------------|------------|----------------------|
| 1 | `NotImplementedError` in KV Cache | Mamba recursive blocks use fixed page size (1) while attention blocks use dynamic page size (16). | Patched `unify_kv_cache_spec_page_size` in `kv_cache_utils.py` to pad and mask mixed geometries. |
| 2 | `AssertionError: M-RoPE support...` | vLLM runner asserts on visual positional embeddings config properties for Qwen3.5-4B. | Patched `uses_mrope` inside `vllm/transformers_utils/config.py` to return `False` for pure-text tasks. |
| 3 | Training steps too low (1–7 steps) | Effective batch size (16) on small datasets with 1 epoch left LoRA adapters untrained. | Updated `run_all_ablation.sh` to scale up DPO and SW-DPO training to **5 epochs** (5, 20, 35 steps). |
| 4 | Volatile accuracy warnings (1.0 or 0.25) | Watchdog loop read training remainder batches (2 or 4 samples) instead of stable evaluation sets. | Recognized logging misattribution in `train_dpo.py` / `train_swdpo.py` and validated healthy status via true `eval_loss`. |
| 5 | Bulky intermediate checkpoints | SFT and DPO training preserved complete `checkpoint-*` directories consuming 8.9 GB. | Implemented automatic recursive post-training directory cleanup across all training backends. |

### Code & Config Changes

| File | Changes |
|------|---------|
| `scripts/run_all_ablation.sh` | Appended `--epochs 5` to all training invocations. |
| `scripts/train_dpo.py` | Added post-training automatic directory cleanup sweep for intermediate checkpoint folders. |
| `scripts/train_swdpo.py` | Added post-training automatic directory cleanup sweep for intermediate checkpoint folders. |
| `scripts/train_sft.py` | Added post-training automatic directory cleanup sweep for intermediate checkpoint folders. |
| `docs/known-issues.md` | Documented Mambaspec/uses_mrope, storage leaks, and DPO watchdog remainder logging behavior. |

### Key Lesson / Thesis Appendix Context

> DPO preference learning on high-quality clinical preference datasets (e.g., golden PhysioNet pairs) is highly sensitive to the combination of effective batch size and gradient budget. When utilizing small datasets with large gradient accumulation parameters, **1 epoch is mathematically insufficient** (equivalent to 1–7 optimizer updates). Scaling training to **5 epochs** guarantees a healthy gradient budget for LoRA updates while preventing overfitting when combined with a conservative learning rate ($5 \times 10^{-6}$).

---

## 2026-05-18: Blackwell DPO Pipeline Debugging & Environment Migration

### Problem

DPO/SW-DPO training scripts (`train_dpo.py`, `train_swdpo.py`) could not run on RTX 5060 Ti (Blackwell, sm_120). Five cascading blockers were encountered and resolved over 3 days.

### Blockers Resolved

| # | Error | Root Cause | Fix |
|---|-------|------------|-----|
| 1 | `OSError: libnvJitLink.so.13` | bitsandbytes JIT needs CUDA 13 linker; hidden in `site-packages/nvidia/cu13/lib/` | Fresh env with native `cuda-toolkit` |
| 2 | TileLang JIT crash | System `/usr/bin/nvcc` too old for sm_120a | `conda install cuda-nvcc` (superseded by env rebuild) |
| 3 | `KeyError: 'images'` | Unsloth VLM detection matched Qwen3.5 as vision model | Pop model_type from `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` |
| 4 | `ValueError: Incorrect image source` | VLM processor routed text through image pipeline | Pass `text_tokenizer` not processor wrapper |
| 5 | `torch.OutOfMemoryError` | DPO logits: `2×4096×152K×4B ≈ 4.7GB` | `precompute_ref_log_probs=True` + `max_seq_length=2048` |

### Environment Migration

| | `vinhthesis` (old) | `vinhthesis2` (new) |
|---|---|---|
| PyTorch | 2.12.0+cu130 | 2.10.0+cu128 |
| CUDA Toolkit | 13.0 (scattered libs) | 12.8 (native conda) |
| bitsandbytes | ❌ JIT fails | ✅ SUCCESS |
| Unsloth | 2026.4.8 | 2026.5.2 |

### Code Changes

| File | Changes |
|------|---------|
| `scripts/train_dpo.py` | Removed Blackwell hacks, added VLM mapping patch + text_tokenizer fix, `precompute_ref_log_probs`, `max_seq_length=2048` |
| `scripts/train_swdpo.py` | Same fixes as train_dpo.py |
| `docs/setup.md` | Added `vinhthesis2` setup, updated env table, DPO VRAM estimates |
| `docs/known-issues.md` | Marked 4 Blackwell issues as resolved, added VLM detection + OOM issues |
| `docs/checklist.md` | Updated Phase 4D status, added Blackwell debugging substeps |

### Key Lesson

> On bleeding-edge hardware (Blackwell sm_120), Python-level `LD_LIBRARY_PATH` modification is too late — the OS dynamic linker reads it only at process start. The correct fix is installing CUDA libraries natively into the conda environment via `conda install -c nvidia cuda-toolkit`.

---

## 2026-05-16: vLLM Inference Pipeline + Blackwell GPU Issues 🚧

### New Files

| File | Purpose |
|------|---------|
| `scripts/merge_lora.py` | Merge SFT LoRA adapter → full bf16 model using Unsloth native merge |
| `scripts/run_sft_inference.py` | vLLM batch inference on 1,500 test samples → predictions.jsonl |
| `scripts/diag_inference.py` | Timing profiler — revealed Unsloth running at 15 tok/s (no FA2/XFormers) |
| `requirements/vllm.txt` | New `vllm` conda env (Python 3.12) setup with cu130 backend |

### Root Cause Analysis: Slow Inference

Profiling with `diag_inference.py` revealed the inference bottleneck:

| Finding | Value |
|---------|-------|
| Unsloth warmup (5 tokens) | 5.28s |
| Generate 512 tokens | 33.45s (15.3 tok/s) |
| GPU utilization | 33% |
| Peak VRAM | 3.73 GB / 15.5 GB |
| Flash Attention | ❌ None (FA2 = False, XFormers = None) |

Root cause: Unsloth prints `"The fast path is not available because one of the required library is not installed"` — neither `causal-conv1d` nor `flash-linear-attention` installed. Model falls back to pure PyTorch attention. Solution: use vLLM (PagedAttention + fused kernels).

### vLLM Inference Pipeline

Two-step process:
1. `scripts/merge_lora.py` — merges QLoRA adapter into full bf16 model (needed because vLLM can't load 4-bit bitsandbytes QLoRA + Unsloth format natively)
2. `scripts/run_sft_inference.py` — vLLM batch inference with PagedAttention

### Blockers Discovered

| Blocker | Description |
|---------|------------|
| `merge_lora.py` | bitsandbytes fails: `libnvJitLink.so.13` not found (CUDA 13.x toolkit not installed) |
| `run_sft_inference.py` | vLLM FlashInfer JIT fails: sm_120 (Blackwell) not in supported arch list |

Both are RTX 5060 Ti (Blackwell, sm_120) + prebuilt wheel incompatibilities. See `docs/known-issues.md` for detailed fix options.

### Design Decisions

| Decision | Reasoning |
|----------|-----------|
| vLLM over Unsloth for inference | vLLM: PagedAttention, ~50-80 tok/s. Unsloth without FA2: ~15 tok/s |
| Merge LoRA before vLLM | vLLM QLoRA bitsandbytes path requires fixed adapter at startup + no dynamic swap |
| `merged_16bit` save method | bf16 compatible with `vllm serve --dtype bfloat16`; avoids re-quantization |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | Disables FlashInfer JIT sampler; falls back to PyTorch multinomial |
| `--dry-run` mode | Validates 1,500 samples load correctly without model — passes ✅ |
| 4th conda env (`vllm`, Python 3.12) | vLLM requires Python 3.10–3.13; `vinhthesis` uses 3.11 (compatible but isolated for safety) |

---

## 2026-05-11: SFT Training Complete (Phase 4C) ✅

### New Files

| File | Purpose |
|------|---------|
| `scripts/extract_sft_data.py` | Extract 30K SFT samples from MIMIC-IV-BHC with quality filters |
| `scripts/train_sft.py` | Production-grade SFT training with crash recovery + persistent logging |
| `data/processed/sft/train_30k.jsonl` | 30,000 pre-extracted training samples |
| `data/processed/sft/extraction_report.json` | Extraction statistics and token distributions |
| `models/qwen35_4b_sft_lora/adapter_model.safetensors` | **163MB LoRA adapter (final model)** |
| `models/qwen35_4b_sft_lora/training_meta.json` | Training metadata (losses, config, timing) |
| `models/qwen35_4b_sft_lora/training_metrics.csv` | Step-level metrics (534 entries) |

### Training Results

| Metric | Value |
|--------|-------|
| Final train loss | **0.973** (from 3.02 at step 10) |
| Final eval loss | **0.942** |
| Train-eval gap | 0.031 (no overfitting) |
| Total steps | 5,346 (3 epochs) |
| Training time | **94.86 hours** (May 7–10) |
| Adapter size | 163MB (safetensors) |

### Architecture Decisions

| Decision | Reasoning |
|----------|-----------|
| 30K samples (not 15K) | LoRA literature consensus: 10-30K optimal for domain adaptation |
| max_seq_length=4096 | 3800 token cap at extraction ensures 100% fit; covers ~96.5% of original dataset |
| Pre-extracted JSONL | Decouples data from training; reproducible; faster startup |
| Unsloth QLoRA (r=32, α=64) | Higher rank for domain adaptation; 2× faster than standard PEFT |
| CSVLoggingCallback | Step-level metrics persisted to disk; survives SSH/crash; easy plotting |
| XFormers (not Flash Attention) | FA2 build fails without CUDA_HOME; XFormers is drop-in, ~10-15% slower |
| Test set exclusion | 1,500 evaluation IDs hardcoded out of 30K training sample pool |
| `per_device_eval_batch_size=1` | **Critical OOM fix**: HF default=8 caused 10.72 GiB allocation during eval |
| batch_size=1, grad_accum=16 | Lower peak VRAM than batch=2×8; same effective batch=16 |
| save_steps=50 | Checkpoints every 50 optimizer steps; survives OOM crashes |

## 2026-05-05: Unsloth Migration (Phase 4B)

### New Files

| File | Purpose |
|------|---------|
| `src/models/unsloth_model.py` | UnslothModel backend — supports inference + LoRA training |
| `configs/models/qwen3_5_4b_unsloth.yaml` | Model config for Unsloth backend |
| `configs/experiment/qwen3_5_4b_unsloth.yaml` | Experiment config for Unsloth inference |
| `scripts/validate_unsloth.py` | Validation: compare Unsloth vs Ollama outputs |
| `requirements/finetune.txt` | Unsloth + TRL + PEFT dependencies |

### Key Findings

- Unsloth loads Qwen3.5-4B in 4-bit NF4 (~4-5 GB VRAM vs ~5 GB Ollama Q8_0)
- Output quality comparable to Ollama (2.03x length ratio = style difference, not quality)
- VL processor workaround needed (Qwen3.5 returns multimodal processor; extract text tokenizer)
- Import order critical: Unsloth MUST load before transformers for monkey-patching

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

## 2026-04-26: CUDA 12.8 Environment

- Driver 575 → 580
- PyTorch cu128 installed
- XFormers used for attention (Flash Attention 2 optional — build requires CUDA_HOME)
- Removed causal-conv1d (flash-linear-attention provides Triton-based conv1d)
- flash_attention_2 fallback in TransformersModel (→ eager if unavailable)

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
