# Reducing Hallucinations in Clinical Text Summarization

Bachelor thesis — LLM-generated clinical text summarization with hallucination reduction techniques.
**Novel contribution**: Severity-Weighted DPO (SW-DPO) — per-sample margins from expert hallucination categories.

## Quick Start

```bash
conda activate vinhthesis

# Baseline (zero-shot):
python scripts/run_experiment.py --config configs/experiment/qwen3_5_2b.yaml --max-samples 2 --range 0_1k

# CoVe (chain-of-verification):
python scripts/run_experiment.py --config configs/experiment/cove/qwen3_5_2b.yaml --max-samples 5 --range 0_1k

# SFT Finetuning:
python scripts/extract_sft_data.py --n-samples 30000           # Extract training data
python scripts/train_sft.py --dry-run                          # Validate data
python scripts/train_sft.py                                     # Run SFT training
python scripts/train_sft.py --resume                            # Resume from checkpoint
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
│   ├── raw/              # MIMIC-IV-BHC CSV (~270K samples)
│   └── processed/
│       ├── mimic_iv_bhc/ # Test set: 1,500 samples (3 ranges × 500)
│       ├── sft/          # SFT training data (train_30k.jsonl + extraction_report.json)
│       └── dpo/          # DPO preference pairs (auto + golden)
├── docs/                 # Documentation
│   ├── setup.md          # Environment setup (3 conda envs + Unsloth)
│   ├── experiments.md    # Running experiments
│   ├── evaluation.md     # Evaluation pipeline
│   ├── architecture.md   # OOP design + finetuning pipeline
│   ├── dpo-implementation.md # Full DPO/SW-DPO plan
│   ├── known-issues.md   # Workarounds and fixes
│   └── checklist.md      # Master progress tracker (START HERE)
├── models/               # Trained model checkpoints
│   └── qwen35_4b_sft_lora/  # SFT LoRA adapter + training logs
├── requirements/         # Dependency files
│   ├── main.txt          # vinhthesis env (inference + eval)
│   ├── finetune.txt      # Unsloth + TRL + PEFT
│   ├── summac.txt        # eval_summac env
│   └── align.txt         # eval_align env
├── src/                  # Source code (Strategy Pattern OOP)
│   ├── data/             # Schema + data loaders
│   ├── models/           # BaseLLM → TransformersModel, OllamaModel, UnslothModel
│   ├── pipelines/        # Summarization + evaluation pipelines
│   ├── prompts/          # Prompt template system
│   ├── techniques/       # BaseTechnique → Baseline, FewShot, CoVe
│   ├── evaluation/       # BaseMetric → completeness + faithfulness
│   └── utils/            # I/O, logging
├── scripts/              # CLI entry points
│   ├── run_experiment.py      # Main inference pipeline
│   ├── run_evaluation.py      # Evaluation pipeline
│   ├── extract_sft_data.py    # SFT data extraction (30K samples)
│   ├── train_sft.py           # SFT training (QLoRA, Unsloth)
│   ├── build_dpo_dataset.py   # Auto-generated DPO pairs (archived)
│   ├── build_golden_dpo_dataset.py  # Expert DPO pairs (100 nested)
│   └── validate_unsloth.py    # Unsloth migration validation
├── tests/                # pytest suite
├── outputs/              # Generated predictions + eval scores
└── pyproject.toml        # Python package config + pytest settings
```

## Documentation

| Doc | Content |
|-----|---------|
| [Checklist](docs/checklist.md) | **Master progress tracker — start here** |
| [Setup](docs/setup.md) | Environment setup, CUDA, conda envs, Unsloth |
| [Architecture](docs/architecture.md) | OOP design, data flow, finetuning pipeline |
| [Experiments](docs/experiments.md) | Running baseline, few-shot, CoVe, CLI params |
| [Evaluation](docs/evaluation.md) | Completeness + faithfulness metrics |
| [DPO Plan](docs/dpo-implementation.md) | Full DPO + SW-DPO methodology |
| [Known Issues](docs/known-issues.md) | Workarounds for BERTScore, deps, BioMistral, etc. |
| [Changelog](docs/changelog.md) | Migration log (vLLM → Transformers → CoVe → Unsloth) |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Inference | HuggingFace Transformers + Ollama + Unsloth |
| Models | BioMistral-7B (variants), Qwen3.5-2B/4B/9B |
| Quantization | bitsandbytes 8-bit / Q8_0 GGUF / NF4 (QLoRA) |
| Techniques | Baseline (zero-shot), Few-Shot (1/5/10), CoVe (3-step), SFT+DPO, **SW-DPO** |
| Finetuning | Unsloth + QLoRA (r=32, α=64) + TRL SFTTrainer/DPOTrainer |
| Evaluation | ROUGE/BLEU/BERTScore/MEDCON + SummaC/AlignScore |
| GPU | NVIDIA RTX 5060 Ti 16GB (Blackwell) |
| PyTorch | 2.10.0+cu128, XFormers attention |
| Environment | conda + uv pip + Python 3.11 |

## Status

- [x] Baseline experiments (5 models × 3 ranges)
- [x] Few-shot experiments (1/5/10-shot × 5 models × 3 ranges)
- [x] Faithfulness evaluation (baseline + few-shot + CoVe)
- [x] CoVe experiments (3 Qwen models × 3 ranges — BioMistral excluded, see [Known Issues](docs/known-issues.md))
- [x] Completeness evaluation (ROUGE/BLEU/BERTScore/MEDCON)
- [x] Golden DPO dataset built (100 expert pairs, nested 10 ⊂ 50 ⊂ 100)
- [x] Unsloth migration complete (UnslothModel backend + validation)
- [x] SFT data extraction (30K samples, test set excluded)
- [x] **SFT training complete** (train_loss=0.973, eval_loss=0.942, 94.86 hrs)
- [ ] SFT model evaluation on 1,500 test set
- [ ] DPO-Uniform training (3-way size ablation: 10/50/100 pairs)
- [ ] **SW-DPO training (3-way severity ablation: Uniform/Binary/Severity)** ⭐
- [ ] Combined SW-DPO + CoVe stacking
- [ ] Results analysis & comparison
