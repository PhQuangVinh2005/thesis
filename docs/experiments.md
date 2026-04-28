# Running Experiments

## CLI Parameters

| Parameter | Description |
|-----------|-------------|
| `--config` | Config file (1 model) |
| `--all` | Run all baseline models |
| `--technique` | Run all models for a technique (`fewshot_1`, `fewshot_5`, `fewshot_10`, `cove`) |
| `--dry-run` | Print prompt, don't load model |
| `--max-samples N` | Limit N samples per range |
| `--range ID` | Only run 1 range (`0_1k`, `1k_2k`, `2k_4k`) |

## Baseline (zero-shot)

```bash
# Single model:
python scripts/run_experiment.py --config configs/experiment/biomistral7b.yaml

# All 5 models:
python scripts/run_experiment.py --all

# Debug (5 samples):
python scripts/run_experiment.py --all --max-samples 5

# Dry-run:
python scripts/run_experiment.py --all --dry-run
```

## Few-Shot (1/5/10-shot)

```bash
python scripts/run_experiment.py --technique fewshot_1
python scripts/run_experiment.py --technique fewshot_5
python scripts/run_experiment.py --technique fewshot_10

# Single model:
python scripts/run_experiment.py --config configs/experiment/fewshot_5/qwen3_5_2b.yaml
```

| Technique | Examples | Selection |
|-----------|----------|-----------|
| `fewshot_1` | 1 | Longest (idx 36) |
| `fewshot_5` | 5 | 5 shortest |
| `fewshot_10` | 10 | 10 shortest |

## Chain-of-Verification (CoVe)

3 LLM calls per sample: draft → plan verification questions → verify+refine.

> **Note**: CoVe runs only on Qwen3.5 models (2B/4B/9B). BioMistral models were excluded due to insufficient instruction-following capability for multi-step verification. See [Known Issues](known-issues.md) for details.

### Quick Test (5 samples)

```bash
python scripts/run_experiment.py --config configs/experiment/cove/qwen3_5_2b.yaml --max-samples 5 --range 0_1k
python scripts/run_experiment.py --config configs/experiment/cove/qwen3_5_4b.yaml --max-samples 5 --range 0_1k
python scripts/run_experiment.py --config configs/experiment/cove/qwen3_5_9b.yaml --max-samples 5 --range 0_1k
```

### Full Run (500 samples × 3 ranges)

```bash
python scripts/run_experiment.py --config configs/experiment/cove/qwen3_5_2b.yaml
python scripts/run_experiment.py --config configs/experiment/cove/qwen3_5_4b.yaml
python scripts/run_experiment.py --config configs/experiment/cove/qwen3_5_9b.yaml
```

### CoVe Pipeline Steps

| Step | Prompt | Model Sees | Output |
|------|--------|-----------|--------|
| 1. Draft | `baseline_summarize` | context | Initial summary (may hallucinate) |
| 2. Plan | `cove_plan` | draft + context | N verification questions |
| 3. Verify+Refine | `cove_verify_refine` | context + questions (no draft) | PART 1: verification + PART 2: corrected summary |

**Option C design**: The verify step does NOT see the draft, preventing the model from merely confirming its own hallucinations.

**Cost**: ~3× inference time vs baseline. `n_questions` configurable in YAML (default: 5).

### CoVe JSONL Output Fields

CoVe predictions include additional fields for thesis analysis:

| Field | Content |
|-------|---------|
| `predicted_summary` | Final corrected summary (extracted from PART 2) |
| `cove_draft` | Step 1 raw draft output |
| `cove_questions` | Step 2 verification questions |
| `cove_raw_verification` | Step 3 full output (PART 1 + PART 2) |
| `instruction` | Trace: `CoVe (n_questions=5): draft(Nc) → plan(Nc) → verify+refine(Nc)` |

## Model Execution Order

| # | Model | Backend | VRAM | Techniques |
|---|-------|---------|------|------------|
| 1 | Qwen3.5-2B | Ollama (Q8_0) | ~2.5 GB | baseline, fewshot, cove |
| 2 | Qwen3.5-4B | Ollama (Q8_0) | ~5 GB | baseline, fewshot, cove |
| 3 | Qwen3.5-9B | Ollama (Q8_0) | ~10 GB | baseline, fewshot, cove |
| 4 | BioMistral-7B | Transformers (8-bit) | ~10 GB | baseline, fewshot |
| 5 | BioMistral-7B-SLERP | Transformers (8-bit) | ~10 GB | baseline, fewshot |

Resume: re-run same command → auto-skips completed samples.
Checkpoint: auto-saves every 10 samples.

## Output Structure

```
outputs/{technique}/{model}/range_{id}/
├── predictions.jsonl       # Generated summaries (+ CoVe intermediates for cove)
├── experiment_meta.json    # Run metadata
├── eval_scores.jsonl       # Completeness per-sample
├── eval_summary.json       # Completeness aggregate
├── summac_scores.jsonl     # SummaC per-sample
├── align_scores.jsonl      # AlignScore per-sample
├── faith_scores.jsonl      # Merged faithfulness
└── faith_summary.json      # Faithfulness aggregate
```

## Config Files

| Path | Purpose |
|------|---------  |
| `configs/experiment/*.yaml` | 1 YAML per model (baseline) |
| `configs/experiment/fewshot_{1,5,10}/*.yaml` | 1 YAML per model × technique |
| `configs/experiment/cove/*.yaml` | CoVe configs (Qwen models only) |
| `configs/models/*.yaml` | Backend, quantization, generation params |
| `configs/prompts/baseline.yaml` | Zero-shot prompt template |
| `configs/prompts/fewshot.yaml` | Few-shot instruction |
| `configs/prompts/cove_plan.yaml` | CoVe Step 2: verification question generation |
| `configs/prompts/cove_verify_refine.yaml` | CoVe Step 3: verify answers + write corrected summary |
| `configs/evaluation.yaml` | Metric params |
