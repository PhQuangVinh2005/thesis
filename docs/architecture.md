# Architecture

## Design Pattern: Strategy Pattern (OOP)

```
BaseLLM (ABC)            → TransformersModel (BioMistral, flash_attn)
                         → OllamaModel (Qwen3.5 via GGUF)
                         → UnslothModel (Qwen3.5-4B via 4-bit NF4, training + inference)

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

### Finetuning Pipeline (SFT → DPO → SW-DPO)

```
Phase 4C — SFT (Domain Adaptation):
  MIMIC-IV-BHC (270K samples)
    → scripts/extract_sft_data.py (filter + sample 30K, exclude test set)
    → data/processed/sft/train_30k.jsonl
    → scripts/train_sft.py (Unsloth QLoRA, chat-formatted)
    → models/qwen35_4b_sft_lora/ (LoRA adapter)

Phase 4D — DPO (Preference Alignment):
  Golden expert pairs (10 ⊂ 50 ⊂ 100)
    → SFT checkpoint as base
    → scripts/train_dpo.py (3-way size ablation)
    → models/qwen35_4b_dpo_uniform_{10,50,100}_lora/

Phase 4D2 — SW-DPO (Novel Contribution):
  Golden pairs + severity weights
    → scripts/train_swdpo.py (3-way severity ablation)
    → models/qwen35_4b_swdpo_{binary,severity}_100_lora/
```

## SFT Training Architecture

```
scripts/extract_sft_data.py              scripts/train_sft.py
┌──────────────────────────┐             ┌──────────────────────────────────┐
│ Load MIMIC-IV-BHC CSV    │             │ Load train_30k.jsonl             │
│ Exclude 1,500 test IDs   │             │ Format chat templates (tqdm)     │
│ Filter tokens (≤3800)    │────────────▶│ Unsloth FastLanguageModel        │
│ Sample 30K (seed=42)     │   JSONL     │ QLoRA (r=32, α=64, all layers)   │
│ Validate + save JSONL    │             │ SFTTrainer (TRL)                 │
└──────────────────────────┘             │ CSVLoggingCallback               │
                                         │ save_strategy=steps (+resume)    │
                                         └──────────────────────────────────┘
                                                     │
                                         models/qwen35_4b_sft_lora/
                                         ├── adapter_model.safetensors
                                         ├── training.log
                                         ├── training_metrics.csv
                                         └── training_meta.json
```

### UnslothModel (`src/models/unsloth_model.py`)

Supports both **inference** and **training**:

- **Inference**: `FastLanguageModel.from_pretrained()` + optional LoRA adapter loading
- **Training**: `get_peft_model()` attaches LoRA adapters for QLoRA finetuning
- **VL Processor Workaround**: Qwen3.5 returns a multimodal processor; model extracts the underlying text tokenizer to avoid image detection crashes on clinical text
- **Import Order**: Unsloth MUST be imported before transformers (monkey-patching)

## Data Schema (4 Core Variables)

| Variable | Field | Meaning |
|----------|-------|---------|
| C | `context` | Raw medical record |
| I | `instruction` | Summarization prompt (or CoVe trace summary) |
| P | `predicted_summary` | AI-generated summary |
| L | `labeled_summary` | Ground truth by physician |

### SFT JSONL Schema (`data/processed/sft/train_30k.jsonl`)

| Field | Type | Meaning |
|-------|------|---------|
| `note_id` | int | MIMIC-IV note identifier |
| `input` | str | Clinical notes (context) |
| `target` | str | Ground truth BHC summary |
| `input_tokens_gpt4` | int | Input token count (GPT-4 tokenizer estimate) |
| `target_tokens_gpt4` | int | Target token count (GPT-4 tokenizer estimate) |

### CoVe JSONL Schema (additional fields)

| Field | Meaning |
|-------|---------|
| `cove_draft` | Step 1: raw baseline draft (may contain hallucinations) |
| `cove_questions` | Step 2: N verification questions generated from draft |
| `cove_raw_verification` | Step 3: full PART 1 (verification) + PART 2 (corrected summary) |
| `instruction` | Trace: `CoVe (n_questions=5): draft(Nc) → plan(Nc) → verify+refine(Nc)` |

## Key Design Decisions

- **Lazy imports** — factory + evaluation use `lambda`/`__getattr__` to avoid heavy deps at import time
- **Unsloth before transformers** — required for monkey-patching optimizations; `train_sft.py` imports Unsloth first inside `train()`, after dry-run exit
- **3 conda envs** — SummaC and AlignScore have irreconcilable version pins
- **Checkpoint + resume** — inference: auto-saves every 10 samples; training: checkpoints every ~500 steps with `--resume` flag
- **Config-driven** — all experiments parameterized via YAML
- **Pre-extracted JSONL** — SFT uses pre-processed data (not raw CSV) for speed + reproducibility
- **Persistent training logs** — `training.log` (text) + `training_metrics.csv` (structured) survive SSH disconnects
- **CoVe Option C** — plan sees draft (to ground questions), verify+refine does NOT see draft (prevents hallucination leakage)
- **HF model fallbacks** — flash_attention_2 → eager, safetensors → pytorch_model.bin (for BioMistral compatibility)

## Dataset: MIMIC-IV-BHC

### Test Set (1,500 samples — NEVER used for training)

500 samples per range (seed=42), 3 ranges by input token length:

| Range | File | Source Population |
|-------|------|-------------------|
| 0-1K | `range_0_1k.jsonl` | 14,193 |
| 1K-2K | `range_1k_2k.jsonl` | 104,637 |
| 2K-4K | `range_2k_4k.jsonl` | 139,217 |

### SFT Training Set (30,000 samples)

Extracted from full dataset (~270K) with filters:
- Test IDs excluded (1,500 samples)
- Target tokens: 50-2000 range
- Total tokens (input + target): ≤ 3,800
- Token stats: input mean=1976, target mean=448, total max=3800

## Models

| Model | Backend | Quantization | VRAM (inference) | VRAM (training) |
|-------|---------|-------------|---------|---------|
| Qwen3.5-2B | Ollama (Q8_0) | GGUF | ~2.5 GB | — |
| Qwen3.5-4B | Ollama (Q8_0) | GGUF | ~5 GB | — |
| Qwen3.5-4B | **Unsloth** | **NF4 (4-bit)** | **~4-5 GB** | **~10-12 GB** |
| Qwen3.5-9B | Ollama (Q8_0) | GGUF | ~10 GB | — |
| BioMistral-7B | Transformers (8-bit) | bitsandbytes | ~10 GB | — |
| BioMistral-7B-SLERP | Transformers (8-bit) | bitsandbytes | ~10 GB | — |

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
