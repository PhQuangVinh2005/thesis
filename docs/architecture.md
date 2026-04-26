# Architecture

## Design Pattern: Strategy Pattern (OOP)

```
BaseLLM (ABC)            → TransformersModel (BioMistral, flash_attn_3)
                         → OllamaModel (Qwen3.5)

BaseDataLoader (ABC)     → MIMICBHCLoader

BaseTechnique (ABC)      → BaselineTechnique (zero-shot)
                         → FewShotTechnique (1/5/10-shot)

BaseMetric (ABC)         → ROUGEMetric, BLEUMetric, BERTScoreMetric, MEDCONMetric
                         → SummaCMetric, AlignScoreMetric
```

**Backend swap** = change 1 line in YAML. **New technique** = new file in `src/techniques/`.

## Data Flow

```
EvalSample → PromptTemplate.format() → Technique.generate(model, prompt) → EvalSample.predicted_summary
```

## Data Schema (4 Core Variables)

| Variable | Field | Meaning |
|----------|-------|---------|
| C | `context` | Raw medical record |
| I | `instruction` | Summarization prompt |
| P | `predicted_summary` | AI-generated summary |
| L | `labeled_summary` | Ground truth by physician |

## Key Design Decisions

- **Lazy imports** — factory + evaluation use `lambda`/`__getattr__` to avoid heavy deps at import time
- **3 conda envs** — SummaC and AlignScore have irreconcilable version pins
- **Checkpoint + resume** — auto-saves every 10 samples, skips completed on restart
- **Config-driven** — all experiments parameterized via YAML

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
