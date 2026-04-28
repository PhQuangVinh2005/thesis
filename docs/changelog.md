# Changelog

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
