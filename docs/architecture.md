# Architecture

## Design Pattern: Strategy Pattern (OOP)

```
BaseLLM (ABC)            → TransformersModel (BioMistral, flash_attn_3)
                         → OllamaModel (Qwen3.5)

BaseDataLoader (ABC)     → MIMICBHCLoader

BaseTechnique (ABC)      → BaselineTechnique (zero-shot)
                         → FewShotTechnique (1/5/10-shot)
                         → CoVeTechnique (draft→plan→verify+refine)

BaseMetric (ABC)         → ROUGEMetric, BLEUMetric, BERTScoreMetric, MEDCONMetric
                         → SummaCMetric, AlignScoreMetric
```

**Backend swap** = change 1 line in YAML. **New technique** = new file in `src/techniques/`.

## Data Flow

### Baseline / Few-Shot

```
EvalSample → PromptTemplate.format() → Technique.generate(model, prompt) → EvalSample.predicted_summary
```

### CoVe (3 LLM calls per sample)

```
EvalSample
  → Step 1: baseline_prompt → model.generate() → draft
  → Step 2: plan_template.format(draft, context) → model.generate() → questions
  → Step 3: verify_template.format(context, questions) → model.generate() → raw_output
  → _extract_summary(raw_output) → EvalSample.predicted_summary

Intermediates stored in EvalSample.metadata:
  - cove_draft              (Step 1 output)
  - cove_questions           (Step 2 output)
  - cove_raw_verification    (Step 3 full output, before extraction)
```

## Data Schema (4 Core Variables)

| Variable | Field | Meaning |
|----------|-------|---------|
| C | `context` | Raw medical record |
| I | `instruction` | Summarization prompt (or CoVe trace summary) |
| P | `predicted_summary` | AI-generated summary |
| L | `labeled_summary` | Ground truth by physician |

### CoVe JSONL Schema (additional fields)

| Field | Meaning |
|-------|---------|
| `cove_draft` | Step 1: raw baseline draft (may contain hallucinations) |
| `cove_questions` | Step 2: N verification questions generated from draft |
| `cove_raw_verification` | Step 3: full PART 1 (verification) + PART 2 (corrected summary) |
| `instruction` | Trace: `CoVe (n_questions=5): draft(Nc) → plan(Nc) → verify+refine(Nc)` |

## Key Design Decisions

- **Lazy imports** — factory + evaluation use `lambda`/`__getattr__` to avoid heavy deps at import time
- **3 conda envs** — SummaC and AlignScore have irreconcilable version pins
- **Checkpoint + resume** — auto-saves every 10 samples, skips completed on restart
- **Config-driven** — all experiments parameterized via YAML
- **CoVe Option C** — plan sees draft (to ground questions in actual claims), verify+refine does NOT see draft (prevents hallucination leakage)
- **HF model fallbacks** — flash_attention_2 → eager, safetensors → pytorch_model.bin (for BioMistral compatibility)

## Dataset: MIMIC-IV-BHC

500 samples per range (seed=42), 3 ranges by input token length:

| Range | File | Source Population |
|-------|------|-------------------|
| 0-1K | `range_0_1k.jsonl` | 14,193 |
| 1K-2K | `range_1k_2k.jsonl` | 104,637 |
| 2K-4K | `range_2k_4k.jsonl` | 139,217 |

## Models

| Model | Backend | Quantization | VRAM |
|-------|---------|-------------|------|
| Qwen3.5-2B | Ollama (Q8_0) | GGUF | ~2.5 GB |
| Qwen3.5-4B | Ollama (Q8_0) | GGUF | ~5 GB |
| Qwen3.5-9B | Ollama (Q8_0) | GGUF | ~10 GB |
| BioMistral-7B | Transformers (8-bit) | bitsandbytes | ~10 GB |
| BioMistral-7B-SLERP | Transformers (8-bit) | bitsandbytes | ~10 GB |

Qwen3.5 uses Ollama because its hybrid DeltaNet architecture needs `flash-linear-attention`.
BioMistral uses standard Transformer architecture — runs directly on HuggingFace Transformers.

### CoVe Model Compatibility

| Model | CoVe Compatible | Reason |
|-------|----------------|--------|
| Qwen3.5-2B/4B/9B | ✅ Yes | Strong instruction-following, proper PART 1/PART 2 structure |
| BioMistral-7B | ❌ No | Token dump drafts, trivial questions (1 instead of 5), no verification structure |
| BioMistral-7B-SLERP | ❌ No | Same issues as base — domain-specific model lacks multi-step reasoning |

## CoVe Extraction Pipeline

`CoVeTechnique._extract_summary()` uses a 6-pattern regex cascade:

1. `### PART 2 — CORRECTED SUMMARY` (markdown header)
2. `Verified/Corrected/Final/Revised Summary:` (bold/plain)
3. `CORRECTED SUMMARY` (all-caps)
4. `PART 2:` (generic marker)
5. `Brief Hospital Course:` (clinical header)
6. Fallback: short output (< 3000 chars) assumed to be summary-only

Verification-only detection: if output starts with `1. CONFIRMED/CONTRADICTED`, logs warning and returns raw output as fallback.
