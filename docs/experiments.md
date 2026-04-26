# Running Experiments

## CLI Parameters

| Parameter | Description |
|-----------|-------------|
| `--config` | Config file (1 model) |
| `--all` | Run all baseline models |
| `--technique` | Run all models for a technique (`fewshot_1`, `fewshot_5`, `fewshot_10`) |
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

## Model Execution Order

| # | Model | Backend | VRAM |
|---|-------|---------|------|
| 1 | Qwen3.5-2B | Ollama (Q8_0) | ~2.5 GB |
| 2 | Qwen3.5-4B | Ollama (Q8_0) | ~5 GB |
| 3 | Qwen3.5-9B | Ollama (Q8_0) | ~10 GB |
| 4 | BioMistral-7B | Transformers (8-bit) | ~10 GB |
| 5 | BioMistral-7B-SLERP | Transformers (8-bit) | ~10 GB |

Resume: re-run same command → auto-skips completed samples.
Checkpoint: auto-saves every 10 samples.

## Output Structure

```
outputs/{technique}/{model}/range_{id}/
├── predictions.jsonl       # Generated summaries
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
|------|---------|
| `configs/experiment/*.yaml` | 1 YAML per model (baseline) |
| `configs/experiment/fewshot_{1,5,10}/*.yaml` | 1 YAML per model × technique |
| `configs/models/*.yaml` | Backend, quantization, generation params |
| `configs/prompts/baseline.yaml` | Zero-shot prompt template |
| `configs/prompts/fewshot.yaml` | Few-shot instruction |
| `configs/evaluation.yaml` | Metric params |
