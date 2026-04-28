# Project Context

> Bachelor thesis: Reducing Hallucinations in Clinical Text Summarization using LLMs.

## Problem

LLMs generate clinically unfaithful content (hallucinations) when summarizing medical records.
This thesis tests hallucination reduction techniques and measures their effectiveness.

## Architecture

**Strategy Pattern OOP** — swap models/techniques/metrics via YAML config.

```
BaseLLM        → TransformersModel (BioMistral)  |  OllamaModel (Qwen3.5)
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

## Environment

| Item | Value |
|------|-------|
| GPU | RTX 5060 Ti 16GB (Blackwell, CUDA 13.0) |
| PyTorch | cu130 + flash_attn_3 |
| Python | 3.11, 3 conda envs |

## Key Paths

| Path | Purpose |
|------|---------  |
| `src/models/` | LLM backends (factory, base, hf_model, ollama_model) |
| `src/techniques/` | Hallucination reduction (base, baseline, fewshot, cove) |
| `src/techniques/cove.py` | CoVe: 3-step draft→plan→verify+refine with regex extraction |
| `src/pipelines/summarizer.py` | Summarization orchestration + CoVe metadata persistence |
| `src/evaluation/` | Metrics (completeness + faithfulness) |
| `src/data/schema.py` | Core data types (EvalSample) |
| `configs/experiment/cove/` | CoVe experiment configs (Qwen only) |
| `configs/prompts/cove_*.yaml` | CoVe prompt templates (plan + verify_refine) |
| `configs/` | YAML configs (models, experiments, prompts, eval) |
| `requirements/` | Dependency files (main, summac, align) |
| `docs/` | Setup, experiments, evaluation, architecture, known issues |
| `scripts/` | CLI entry points (run_experiment, run_evaluation, setup_env) |

## Conventions

- `uv pip install` for main env (except PyTorch cu130 + flash_attn_3)
- Config-driven: swap model/prompt/technique via YAML
- Lazy imports for heavy deps (torch, evaluation metrics)
- 3 conda envs due to irreconcilable SummaC/AlignScore deps
- CoVe intermediate outputs (draft, questions, raw verification) persisted in JSONL metadata
- CoVe only runs on Qwen models (BioMistral lacks instruction-following for multi-step verification)
- HF model loader gracefully falls back: flash_attention_2 → eager, safetensors → pytorch_model.bin
