# Composite Faithfulness Score — Weight Analysis

> Analysis date: 2026-04-29
> Data: 1,500 paired scores (5 models × 3 ranges × 100 samples each from baseline evaluation)

## Formula

```
composite = 0.6 × AlignScore + 0.4 × SummaC_conv
```

## Metric Properties

| Property | AlignScore | SummaC_conv |
|----------|-----------|-------------|
| Mean | 0.587 | 0.430 |
| Std | 0.163 | 0.141 |
| Range | [0.006, 0.992] | [0.210, 0.999] |
| CV (std/mean) | 0.277 | 0.327 |
| Model spread (max−min of model means) | 0.141 | 0.093 |

## Key Finding: Weak Correlation (r = 0.29)

The two metrics have **weak Pearson correlation** (r = 0.285), meaning they capture **different aspects** of faithfulness:

- **AlignScore** (Zha et al., EMNLP 2023): NLI-based, checks if the summary is *entailed* by the source document. Trained specifically for factual consistency evaluation. Sensitive to hallucinated facts.
- **SummaC_conv** (Laban et al., TACL 2022): NLI-based with convolutional aggregation over sentence pairs. Better at catching *localized* contradictions within specific sentences.

This weak correlation is actually **desirable** — combining them captures more hallucination signal than either metric alone.

## Model Ranking Agreement

When picking the best/worst model per sample (across all 5 models):

| Agreement type | Rate |
|---------------|------|
| Best model agrees | 38.3% (115/300) |
| Worst model agrees | 43.7% (131/300) |
| **Best model disagrees** | **61.7%** |
| **Worst model disagrees** | **56.3%** |

→ The metrics **disagree** on which model is most/least faithful in >56% of samples. A composite score arbitrates these disagreements.

## Weight Sensitivity Analysis

| Weights (Align / SummaC) | Same chosen+rejected as 60/40 |
|---------------------------|-------------------------------|
| 50/50 | 87.3% |
| **60/40** (chosen) | **100%** (reference) |
| 70/30 | 87.7% |
| Equal-contribution (46/54) | 83.3% |

**Conclusion: weight choice has low sensitivity.** ~87% of DPO preference pairs are identical regardless of whether 50/50, 60/40, or 70/30 is used. The exact ratio is not critical for DPO pair selection.

## Why 60/40? (Rationale)

The 60/40 weight slightly favoring AlignScore is justified by:

1. **Wider effective range** — AlignScore uses nearly the full [0, 1] scale (0.006–0.992) vs SummaC (0.210–0.999). Scores near 0 from AlignScore indicate strong unfaithfulness; SummaC never goes below 0.21.

2. **Higher model discriminative power** — AlignScore's model spread (0.141) is 1.5× larger than SummaC's (0.093), meaning it better distinguishes quality differences between models.

3. **Literature support** — AlignScore was designed specifically for factual consistency evaluation and shows higher correlation with human judgments than SummaC on summarization benchmarks (Zha et al., 2023, Table 2: AlignScore achieves state-of-the-art on SummaC benchmark itself).

4. **Pragmatic choice** — Given the low weight sensitivity (87%+ agreement across all tested ratios), the exact ratio is less important than using *both* metrics. 60/40 is a reasonable default.

## Per-Model Faithfulness (Baseline Means)

| Model | AlignScore | SummaC_conv | Expected DPO Role |
|-------|-----------|-------------|-------------------|
| biomistral7b | 0.483 | 0.444 | 🔴 Rejected source |
| biomistral7b_slerp | 0.598 | 0.435 | 🟠 Rejected source |
| qwen3_5_2b | 0.617 | 0.479 | 🟡 Mixed |
| qwen3_5_4b | 0.614 | 0.385 | 🟢 On-policy source |
| qwen3_5_9b | 0.624 | 0.406 | 🟢 Chosen source |

> **Note**: SummaC_conv does not rank models the same as AlignScore (e.g., qwen3_5_2b has highest SummaC but mid-range AlignScore). This confirms the composite is needed.

## References

- Zha et al., "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function", EMNLP 2023, arXiv:2305.16739
- Laban et al., "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization", TACL 2022, arXiv:2111.09525
