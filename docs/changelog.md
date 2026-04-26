# Changelog

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
