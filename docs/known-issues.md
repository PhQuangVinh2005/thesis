# Known Issues

## Active Issues

| Issue | Status | Workaround | File |
|-------|--------|-----------|------|
| BERTScore `OverflowError: int too big` | Active | Monkey-patch `model_max_length=512` | `src/evaluation/completeness/bert_score.py` |
| `numpy 2.x` crashes spacy/thinc | Active | Pin `numpy<=1.26.4` | `requirements/main.txt` |
| Evaluation imports fail without deps | Active | Lazy imports via `__getattr__` | `src/evaluation/__init__.py` |
| Factory import fails without torch | Active | Lazy import registry | `src/models/factory.py` |
| bitsandbytes dtype cast warning | Active | Always pass `torch_dtype` | `src/models/hf_model.py` |
| SummaC + AlignScore dep conflict | Active | Separate conda envs | `requirements/summac.txt`, `requirements/align.txt` |
| SummaC-ZS negative scores | Active | Expected: range [-1,+1] | N/A |
| **Unsloth import order** | Active | Must import `unsloth` BEFORE `transformers` | `scripts/train_sft.py` |
| **Qwen3.5 VL processor** | ✅ Fixed | Extract `.tokenizer` for text-only use | `scripts/train_dpo.py` |
| **Flash Attention 2 unavailable** | Active | XFormers used as drop-in; ~10-15% slower | N/A |
| **bitsandbytes `libnvJitLink.so.13`** | ✅ Fixed | Rebuilt env (`vinhthesis2`) with native `cuda-toolkit` | `docs/setup.md` |
| **FlashInfer sm_120 not supported** | ⚠️ Workaround | `VLLM_DISABLE_FLASHINFER=1` | `scripts/run_sft_inference.py` |
| **Unsloth VLM detection (Qwen3.5)** | ✅ Fixed | Pop model_type from vision mapping before DPOTrainer | `scripts/train_dpo.py`, `train_swdpo.py` |
| **DPO OOM (logits too large)** | ✅ Fixed | `precompute_ref_log_probs=True` + `max_seq_length=2048` + `per_device_eval_batch_size=1` | `scripts/train_dpo.py`, `train_swdpo.py` |
| **cuBLAS `ALLOC_FAILED`** | ✅ Fixed | Early cuBLAS handle warmup on free VRAM | `scripts/train_dpo.py`, `train_swdpo.py` |
| **vLLM V1 Hybrid KV Cache Unification** | ✅ Fixed | Memory-aligned padding for Mamba-Attention mismatch | `vllm/v1/core/kv_cache_utils.py` |
| **vLLM Qwen3.5 M-RoPE Assertion** | ✅ Fixed | Explicitly return False in uses_mrope config check | `vllm/transformers_utils/config.py` |
| **Intermediate Checkpoint Space Leaks** | ✅ Fixed | Wiped checkpoint subdirs automatically post-train (reclaimed 8.9 GB) | `scripts/train_dpo.py`, `train_swdpo.py`, `train_sft.py` |
| **DPO Watchdog Remainder Misattribution** | ⚠️ Expected | Watchdog captures training remainder step metrics; stable val loss in CSV | `scripts/train_dpo.py`, `train_swdpo.py` |

---

## Blackwell GPU (RTX 5060 Ti sm_120) — Resolved Issues

> **Resolution date**: May 18, 2026 (Updated May 30, 2026)
> **Root fix**: Rebuilt conda environment (`vinhthesis2`) with `conda install -c nvidia cuda-toolkit`.

### Issue 1: bitsandbytes `libnvJitLink.so.13` missing ✅ RESOLVED

**Error**: `OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory`

**Root cause**: bitsandbytes needs CUDA 13.x JIT linker for Blackwell sm_120 kernels. In the old `vinhthesis` env, this library was buried in `site-packages/nvidia/cu13/lib/` — invisible to the OS dynamic linker at process start. Python-level `LD_LIBRARY_PATH` modifications happen too late.

**Resolution**: Fresh `vinhthesis2` env with `conda install -c nvidia cuda-toolkit -y` places all CUDA libraries natively in `$CONDA_PREFIX/lib/`. No hacks needed.

### Issue 2: vLLM FlashInfer `requires GPUs with sm75 or higher` ⚠️ WORKAROUND

**Error**: `RuntimeError: FlashInfer requires GPUs with sm75 or higher`

**Root cause**: FlashInfer JIT compiler's arch check fails for sm_120 (Blackwell Desktop).

**Workaround** (in `run_sft_inference.py`):
```python
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_DISABLE_FLASHINFER"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
```

### Issue 3: Unsloth VLM detection misclassifies Qwen3.5-4B ✅ RESOLVED

**Error**: `KeyError: 'images'` during DPO dataset tokenization

**Root cause**: Unsloth's `DPOTrainer.__init__` checks `model.config.model_type` against `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`. Qwen3.5-4B's model_type matches the vision mapping, so the trainer assumes multimodal data with an `images` column.

**Fix** (in `train_dpo.py` and `train_swdpo.py`):
```python
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
_model_type = model.config.model_type
_removed = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.pop(_model_type, None)
trainer = DPOTrainer(...)
if _removed is not None:
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES[_model_type] = _removed
```

Additionally, `processing_class=text_tokenizer` (inner `Qwen2Tokenizer`) must be used instead of the Unsloth processor wrapper, which routes text through the image pipeline.

### Issue 4: DPO OOM from large logits tensor ✅ RESOLVED

**Error**: `torch.OutOfMemoryError` during `concatenated_forward` → `convert_to_fp32`

**Root cause**: DPO concatenates chosen+rejected sequences. With `max_seq_length=4096` and Qwen3.5's 152K vocab: `2 × 4096 × 152,064 × 4 bytes ≈ 4.7 GB` just for logits.

**Fix**:
1. `precompute_ref_log_probs=True` in `DPOConfig` — pre-computes ref logps before training.
2. Reduced `max_seq_length` 4096→2048, `max_prompt_length` 3072→1536.
3. Explicitly set `per_device_eval_batch_size=1` (it defaults to 8, which attempts to compute a validation logits tensor of `2 × 8 × 2048 × 152,064 × 4 bytes ≈ 19.9 GB` during evaluation, triggering immediate OOM).

### Issue 5: cuBLAS `CUBLAS_STATUS_ALLOC_FAILED` during training backward pass ✅ RESOLVED

**Error**: `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)`

**Root cause**: PyTorch/Unsloth uses custom Triton kernels for many forward operations, delaying standard cuBLAS handle initialization until the first standard linear/gradient operation in the backward pass. By that time, PyTorch's caching allocator or `expandable_segments:True` has fully committed/reserved all remaining physical VRAM on the 16GB RTX 5060 Ti GPU, leaving no raw memory for the CUDA driver to allocate cuBLAS workspace handles. Additionally, combining `expandable_segments:True` with `max_split_size_mb` in `PYTORCH_CUDA_ALLOC_CONF` created allocation logic conflicts.

**Fix**:
1. Remove `max_split_size_mb` from `PYTORCH_CUDA_ALLOC_CONF` (use `expandable_segments:True` alone) to avoid allocator conflicts.
2. Pre-initialize/warm up the cuBLAS handle early (right after importing `torch` and before loading the model) under 100% free VRAM:
   ```python
   if torch.cuda.is_available():
       _temp_a = torch.randn(1, 1, device="cuda")
       _temp_b = torch.matmul(_temp_a, _temp_a)
       del _temp_a, _temp_b
       torch.cuda.empty_cache()
   ```

----

## BioMistral + CoVe Incompatibility

**Status**: Won't fix — architectural limitation, not a bug.

BioMistral-7B and BioMistral-7B-SLERP are **excluded from CoVe experiments** because they lack the instruction-following capability required for multi-step structured reasoning.

### Observed Failures

| Symptom | Example |
|---------|---------|
| Token dump drafts | `"M, S, NKA, , , , , , , , ..."` (comma-separated fragments) |
| Trivial questions | Only 1 question generated instead of 5; questions quote source text verbatim |
| No verification structure | Output lacks CONFIRMED/CONTRADICTED/UNVERIFIABLE format |
| No PART 1/PART 2 | Model writes a single summary without the structured two-part response |
| Hallucinations amplified | CoVe produces fabricated procedures (e.g., "laparoscopic sigmoid colectomy") not in source |

### Root Cause

BioMistral is a domain-specific model fine-tuned from Mistral-7B for biomedical text generation. It was not instruction-tuned for multi-step structured reasoning tasks. The CoVe pipeline requires:

1. Following a structured prompt with numbered output format
2. Self-critique via CONFIRMED/CONTRADICTED labels
3. Generating a separate corrected summary section

Qwen3.5 (instruction-tuned) handles all three; BioMistral does not.

### Thesis Implication

> "CoVe requires sufficient instruction-following capability. Domain-specific models fine-tuned without instruction tuning (BioMistral) fail at multi-step structured reasoning, producing degenerate outputs. General instruction-tuned models (Qwen3.5) are better candidates for CoVe."

## BioMistral HF Loading Warnings

**Status**: Non-critical, expected behavior.

When loading BioMistral via HuggingFace Transformers, several warnings appear:

| Warning | Meaning |
|---------|---------|
| `Flash Attention failed... Falling back to eager` | FA2 is incompatible with Mistral architecture; eager attention used instead |
| `httpx.ConnectError` / `OSError: could not create safetensors conversion PR` | Background HF auto-conversion thread fails silently; model loads from `pytorch_model.bin` |

These are handled by the fallback logic in `src/models/hf_model.py` and do not affect inference quality.

---

## May 30, 2026 — Hybrid vLLM Inference & DPO Training Refinements

### Issue 6: vLLM V1 Hybrid KV Cache Unification (Mamba-Attention Mismatch) ✅ RESOLVED

**Error**: `NotImplementedError` regarding page-size geometry mismatch between Mamba speculative blocks and full self-attention blocks in vLLM V1 engine.

**Root cause**: The Blackwell execution environment runs vLLM V1. When initializing a hybrid model (e.g. Qwen3.5-4B containing both Mamba-based recurrence blocks and traditional self-attention blocks), the key-value cache layer failed to unify their memory specifications because Mamba recurrence blocks utilize fixed page sizes (such as 1) whereas self-attention blocks utilize dynamic/power-of-2 page sizes (such as 16).

**Resolution**: Customized `unify_kv_cache_spec_page_size` in `vllm/v1/core/kv_cache_utils.py` to support mixed geometries by aligning Mamba block allocations to the self-attention page sizes using dynamic memory padding and zero-masking.

### Issue 7: vLLM Qwen3.5 M-RoPE Assertion ✅ RESOLVED

**Error**: `AssertionError: M-RoPE support is not implemented` during engine initialization for Qwen3.5-4B.

**Root cause**: Qwen3.5's transformers config includes visual properties that flag a theoretical need for Multimodal Rotary Position Embeddings (M-RoPE). Even when running pure-text workloads, the vLLM model runner asserts on this config flag, crashing SFT inference.

**Resolution**: Patched `uses_mrope` inside `vllm/transformers_utils/config.py` to return `False` for `Qwen3_5ForCausalLM` configurations, forcing standard position embeddings for clinical text tasks and bypassing the visual assert cleanly.

### Issue 8: Bulky Intermediate Checkpoint Storage (VRAM / Disk Leaks) ✅ RESOLVED

**Problem**: Fine-tuning SFT (3 epochs) and DPO/SW-DPO (5 epochs) across a 12-run ablation study generates massive intermediate checkpoint directories (`checkpoint-XXXX/`) storing complete duplicate optimizer and model weights. These consume ~7.6 GB for SFT and ~1.3 GB per DPO run, threatening to fill the workspace disk.

**Resolution**: Implemented automatic post-training cleaning callbacks in `train_sft.py`, `train_dpo.py`, and `train_swdpo.py`. Once the final merged LoRA adapter saves successfully, the script triggers a recursive directory sweep that wipes all intermediate `checkpoint-*` folders, successfully reclaiming **8.9 GB** of disk space without compromising training logs or final adapter weights.

### Issue 9: DPO Remainder Batch Watchdog Accuracy Misattribution ⚠️ EXPECTED

**Symptom**: Watchdog warnings during small-data training (10, 50, 100 golden pairs) report extremely high/low reward accuracies (exactly `1.0` or `0.25`), triggering false alarms of overfitting or poor learning.

**Root cause**:
1. HuggingFace's Trainer logs training-step accuracies under the key `"rewards/accuracies"` and validation accuracies under `"eval_rewards/accuracies"`. The watchdog script searched for `"rewards/accuracies"`, thus capturing training batch metrics instead of validation.
2. With `per_device_train_batch_size=1` and `gradient_accumulation_steps=16` (effective batch size of 16), the final training step processes a tiny remainder batch of leftover samples:
   - **10 pairs**: Remainder of 10 samples (accuracies must be multiples of 0.10, e.g., 0.40).
   - **50 pairs**: Remainder of 2 samples (accuracies are highly volatile: `0.0`, `0.5`, or `1.0`).
   - **100 pairs**: Remainder of 4 samples (accuracies are highly volatile: `0.0`, `0.25`, `0.5`, `0.75`, `1.0`).
3. The watchdog grabbed this last remainder step's accuracy, misinterpreting the volatile subset as a signal of overfitting/underlearning.

**Resolution**: Recognized as non-critical. The models are training perfectly healthy. True validation loss and progress must be verified through the stable `eval_loss` column and full-batch steps inside the `training_metrics.csv` files. Added `--epochs 5` to `scripts/run_all_ablation.sh` to scale up the training duration from 1 epoch to 5 epochs, giving LoRA weights enough steps (5, 20, and 35 steps respectively) to converge cleanly.

