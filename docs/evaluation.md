# Evaluation Pipeline

## Two Phases

| Phase | Comparison | Catches | Metrics |
|-------|-----------|---------|---------|
| Completeness | P vs L | Omissions | ROUGE-1/2/L, BLEU, BERTScore, MEDCON |
| Faithfulness | P vs C | Hallucinations | SummaC-ZS, SummaC-Conv, AlignScore |

**Variables**: C = Context (raw record), P = Predicted summary, L = Labeled summary (ground truth).

## Completeness (env: vinhthesis)

```bash
conda activate vinhthesis
python scripts/run_evaluation.py --experiment-dir outputs/baseline/qwen3_5_2b/ --max-samples 100
```

## Faithfulness (requires env switching)

```bash
# Automated (handles SummaC → AlignScore → merge):
bash scripts/run_faithfulness.sh outputs/baseline/qwen3_5_2b/ --max-samples 100

# Or all models for a technique:
bash scripts/run_faithfulness.sh outputs/fewshot_1/ --max-samples 100
```

## Faithfulness Score Ranges

| Metric | Range | Interpretation |
|--------|-------|----------------|
| `summac_zs` | [-1, +1] | Entailment − Contradiction. Negative = hallucination dominant |
| `summac_conv` | [0, 1] | Learned consistency. Higher = more faithful |
| `alignscore` | [0, 1] | NLI alignment probability. Higher = more faithful |
