# Reducing Hallucinations in Clinical Text Summarization

Bachelor thesis — LLM-generated clinical text summarization with hallucination reduction techniques.

## Quick Start

```bash
conda activate vinhthesis

# Baseline (zero-shot):
python scripts/run_experiment.py --config configs/experiment/qwen3_5_2b.yaml --max-samples 2 --range 0_1k

# CoVe (chain-of-verification):
python scripts/run_experiment.py --config configs/experiment/cove/qwen3_5_2b.yaml --max-samples 5 --range 0_1k
```

## Project Structure

```
thesis/
├── configs/              # YAML configs (models, experiments, prompts, eval)
│   ├── experiment/       # Baseline + technique experiment configs
│   │   ├── *.yaml        # Baseline (1 per model)
│   │   ├── fewshot_{1,5,10}/ # Few-shot configs
│   │   └── cove/         # CoVe configs (Qwen only)
│   ├── models/           # Model backend configs
│   └── prompts/          # Prompt templates (baseline, fewshot, cove_plan, cove_verify_refine)
├── data/                 # Raw + preprocessed datasets
├── docs/                 # Documentation
│   ├── setup.md          # Environment setup (3 conda envs)
│   ├── experiments.md    # Running experiments
│   ├── evaluation.md     # Evaluation pipeline
│   ├── architecture.md   # OOP design + project layout
│   ├── known-issues.md   # Workarounds and fixes
│   └── changelog.md      # Migration history
├── requirements/         # Dependency files
│   ├── main.txt          # vinhthesis env
│   ├── summac.txt        # eval_summac env
│   └── align.txt         # eval_align env
├── src/                  # Source code (Strategy Pattern OOP)
│   ├── data/             # Schema + data loaders
│   ├── models/           # BaseLLM → TransformersModel, OllamaModel
│   ├── pipelines/        # Summarization + evaluation pipelines
│   ├── prompts/          # Prompt template system
│   ├── techniques/       # BaseTechnique → Baseline, FewShot, CoVe
│   ├── evaluation/       # BaseMetric → completeness + faithfulness
│   └── utils/            # I/O, logging
├── scripts/              # CLI entry points
├── tests/                # pytest suite
├── outputs/              # Generated predictions + eval scores
└── pyproject.toml        # Python package config + pytest settings
```

## Documentation

| Doc | Content |
|-----|---------|
| [Setup](docs/setup.md) | Environment setup, CUDA, conda envs |
| [Experiments](docs/experiments.md) | Running baseline, few-shot, CoVe, CLI params |
| [Evaluation](docs/evaluation.md) | Completeness + faithfulness metrics |
| [Architecture](docs/architecture.md) | OOP design, data flow, extensibility |
| [Known Issues](docs/known-issues.md) | Workarounds for BERTScore, deps, BioMistral, etc. |
| [Changelog](docs/changelog.md) | Migration log (vLLM → Transformers → CoVe) |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Inference | HuggingFace Transformers + Ollama |
| Models | BioMistral-7B (variants), Qwen3.5-2B/4B/9B |
| Quantization | bitsandbytes 8-bit / Q8_0 GGUF |
| Techniques | Baseline (zero-shot), Few-Shot (1/5/10), CoVe (3-step) |
| Evaluation | ROUGE/BLEU/BERTScore/MEDCON + SummaC/AlignScore |
| GPU | NVIDIA RTX 5060 Ti 16GB (Blackwell, CUDA 13.0) |
| PyTorch | cu130, Flash Attention 3 |
| Environment | conda + uv pip + Python 3.11 |

## Status

- [x] Baseline experiments (5 models × 3 ranges)
- [x] Few-shot experiments (1/5/10-shot × 5 models × 3 ranges)
- [x] Faithfulness evaluation (baseline)
- [ ] CoVe experiments (3 Qwen models × 3 ranges — BioMistral excluded, see [Known Issues](docs/known-issues.md))
- [ ] Faithfulness evaluation (few-shot)
- [ ] Faithfulness evaluation (CoVe)
- [ ] Completeness evaluation
- [ ] Results analysis & comparison (Baseline vs Few-Shot vs CoVe)
