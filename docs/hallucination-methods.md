# Hallucination Reduction Methods — Decision Log

> Research-backed methods evaluated for reducing hallucination in clinical text summarization.
> Each method cites its source paper. Status indicates whether it was selected for implementation.

## Overview

```
Methods evaluated: 14 total
  ├── Implemented (3):  Baseline, Few-Shot, CoVe
  ├── Implementing (2): DPO (baseline), SW-DPO (core contribution)
  ├── Selected (3):     Multi-Agent Debate, DoLa, Summarize-then-Prompt
  └── Rejected (6):     Self-Refine, CAD, NER Grounding, GRPO, SPIN, KTO*
                        (* KTO kept as fallback for DPO)
```

---

## Tier 1: Prompt-Level Methods (Inference-Time)

### 1. Chain-of-Verification (CoVe) — ✅ IMPLEMENTED

- **Paper**: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in Large Language Models", arXiv:2309.11495, 2023.
- **Status**: Implemented, Qwen3.5 only (BioMistral cannot follow multi-step instructions)
- **Mechanism**: Draft → Plan verification questions → Verify+Refine without seeing draft
- **Design choice**: Option C — plan step sees draft (to ground questions), verify+refine does NOT see draft (prevents hallucination leakage)
- **LLM calls**: 3× per sample
- **Location**: `src/techniques/cove.py`, `configs/experiment/cove/`

### 2. Self-Refine (Iterative Self-Feedback) — ❌ SKIPPED

- **Paper**: Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback", NeurIPS 2023, arXiv:2303.17651.
- **Decision**: Skip. CoVe is more structured (explicit verification questions vs free-form critique). Both cost ~3× LLM calls, but CoVe forces systematic checking. Since CoVe is already implemented and tested, adding Self-Refine would not provide a fundamentally different mechanism.
- **LLM calls**: 2-3× per sample

### 3. Multi-Agent Self-Debate — 📋 SELECTED (future)

- **Paper**: Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate", ICML 2024, arXiv:2305.14325.
- **Also**: Liang et al., "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate", ACL 2024, arXiv:2305.19118.
- **Decision**: Keep. Addresses Degeneration-of-Thought problem (model confirming its own errors). Multiple perspectives catch different hallucination types. Need to design a cost-reduced version before implementation.
- **Concern**: Expensive (6-9× LLM calls). Will run on a smaller subset.
- **LLM calls**: 6-9× per sample

### 4. Atomic Self-Consistency (ASC) — ❌ NOT SELECTED

- **Paper**: Thirukovalluru et al., "Atomic Self-Consistency for Better Long Form Generations", EMNLP 2024, arXiv:2405.13131.
- **Decision**: Not prioritized. Interesting concept (hallucinated facts vary across samples, real facts are consistent), but replaced by the DPO finetuning approach which addresses the root cause rather than post-hoc filtering.

---

## Tier 2: Decoding-Level Methods (Requires HF Transformers)

### 5. DoLa (Decoding by Contrasting Layers) — 📋 SELECTED (idea)

- **Paper**: Chuang et al., "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models", ICLR 2024, arXiv:2309.03883.
- **Decision**: Keep as an idea. If Qwen moves to HF Transformers for DPO finetuning, DoLa is a 1-line change: `model.generate(..., dola_layers="high", repetition_penalty=1.2)`. Zero extra LLM calls.
- **Constraint**: Only works with HF Transformers (not Ollama). Focus is Qwen family, so depends on the HF migration in Phase C of DPO plan.
- **LLM calls**: 1× (same as baseline — just modifies decoding)

### 6. Context-Aware Decoding (CAD) — ❌ SKIPPED

- **Paper**: Shi et al., "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding", NAACL 2024, arXiv:2305.14739.
- **Decision**: Skip. Requires per-token logit access (2× inference per token: with context vs without context). Conceptually strong for clinical summarization (forces model to trust context over priors), but implementation complexity is high and it doubles inference time. DoLa is simpler for a similar effect.
- **LLM calls**: 2× per token

---

## Tier 3: Input-Level Methods (Preprocessing)

### 7. NER Entity Grounding — ❌ SKIPPED

- **Paper**: Agrawal et al., "Large Language Models are Few-Shot Clinical Information Extractors", EMNLP 2022, arXiv:2205.12689.
- **Decision**: Skip. Depends too much on the quality of the medical entity database. SciSpaCy NER (`en_core_sci_lg`) is used for MEDCON evaluation but its entity extraction is not reliable enough to serve as ground-truth constraints for prompting.

### 8. Summarize-then-Prompt (Hierarchical Summarization) — 📋 NOTED

- **Paper**: Van Veen et al., "Adapted Large Language Models Can Outperform Medical Experts in Clinical Text Summarization", Nature Medicine, 2024.
- **Decision**: Noted for future work. Particularly useful for the 2K-4K token range where hallucination rates are highest. Would split clinical notes by section headers → summarize each → merge.
- **LLM calls**: N+1× (N sections + 1 merge)

---

## Tier 4: Finetuning Methods

### 9. SFT + DPO (Two-Stage Pipeline) — ✅ BASELINE FOR SW-DPO

- **Papers**:
  - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023, arXiv:2305.14314.
  - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022, arXiv:2106.09685.
  - Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023, arXiv:2305.18290.
  - Tian et al., "Fine-tuning Language Models for Factuality", ICLR 2024, arXiv:2311.08401.
- **Evidence**: 40% reduction in medical hallucination (Tian et al.)
- **Decision**: Serves as the ablation baseline for SW-DPO (Method 14). Two stages:
  1. **SFT**: Teach clinical format using MIMIC-IV-BHC ground truth (context → labeled_summary)
  2. **DPO**: Align faithfulness using golden expert pairs from Hegselmann et al.
- **Hardware**: Qwen3.5-4B + QLoRA on RTX 5060 Ti 16GB via Unsloth
- **Dataset**: 100 expert-annotated golden pairs (Hegselmann et al., CHIL 2024). Nested 10 ⊂ 50 ⊂ 100 ablation.
- **Full plan**: See `docs/dpo-implementation.md`

### 14. Severity-Weighted DPO (SW-DPO) — ⭐ CORE CONTRIBUTION, IMPLEMENTING

- **Papers**:
  - Chaduvula et al., "Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning" (F-DPO), arXiv:2601.03027, 2026.
  - Hegselmann et al., "Data-Centric Approach to Reducing Clinical Hallucinations", CHIL 2024, arXiv:2402.15422.
  - NCC MERP, "Index for Categorizing Medication Errors", 2001. (Clinical severity taxonomy)
- **Evidence**: F-DPO shows factuality-conditioned margins significantly improve DPO for hallucination reduction.
- **Core idea**: Modify DPO loss with **per-sample severity margins** derived from expert-annotated hallucination categories. Clinically dangerous errors (medication fabrication, contradicted facts) receive larger gradient push than benign word-level variations.
- **What makes it novel**: Hegselmann et al. used the 11-category annotations only for analysis; we use them as a **training signal** by mapping categories to clinical severity weights via the NCC MERP patient safety framework.
- **Severity weights** (derived from NCC MERP, no human annotation needed):
  - `contradicted_fact`: 5.0 | `medication_unsupported`: 4.0
  - `condition_unsupported`: 3.5 | `procedure_unsupported`: 3.0 | `number_unsupported`: 3.0
  - `time_unsupported`: 2.0 | `location_unsupported`: 2.0
  - `name_unsupported`: 1.5 | `word_unsupported`: 1.0 | `other_unsupported`: 1.0
- **Loss modification**: `L = -log σ(β · (log-ratio_chosen - log-ratio_rejected) - α · norm_severity)`
- **3-way ablation**:
  1. DPO-Uniform (standard β, no margin) — replication baseline
  2. SW-DPO-Binary (margin = 0 or 1) — any-severity control
  3. SW-DPO-Severity (margin = normalized severity score) — full contribution
- **Full plan**: See `docs/dpo-implementation.md` Phase E2

### 10. ORPO (Monolithic Preference Optimization) — 📋 BACKUP

- **Paper**: Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model", EMNLP 2024, arXiv:2403.07691.
- **Decision**: Backup if DPO causes OOM (ORPO needs no reference model → saves ~2GB VRAM). Single-stage (SFT + alignment in one step). Less proven for factuality specifically.

### 11. KTO (Kahneman-Tversky Optimization) — 📋 FALLBACK

- **Paper**: Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", ICML 2024, arXiv:2402.01306.
- **Decision**: Fallback if DPO preference pairs are too noisy (AlignScore/SummaC can't reliably distinguish good from bad). KTO only needs binary labels (good/bad), not paired preferences.

### 12. GRPO (Group Relative Policy Optimization) — ❌ SKIPPED

- **Paper**: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024, arXiv:2402.03300.
- **Decision**: Skip. Primarily validated on math/reasoning, not summarization. Needs K samples during training = high VRAM pressure on 16GB. More experimental than DPO for this use case.

### 13. SPIN (Self-Play Fine-Tuning) — ❌ SKIPPED

- **Paper**: Chen et al., "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models", ICML 2024, arXiv:2401.01335.
- **Decision**: Skip. General quality improvement, not specifically designed for hallucination reduction. DPO with faithfulness-specific preference pairs is a more targeted approach.

---

## Method Selection Summary

| Method | Category | Status | Reason |
|--------|----------|--------|--------|
| Baseline | Prompt | ✅ Done | Control |
| Few-Shot | Prompt | ✅ Done | Standard technique |
| CoVe | Prompt | ✅ Done (Qwen) | Structured verification, proven |
| Multi-Agent Debate | Prompt | 📋 Future | Novel but expensive |
| DoLa | Decoding | 📋 Idea | 1-line change if on HF |
| Summarize-then-Prompt | Input | 📋 Noted | Good for long inputs |
| SFT → DPO (Uniform) | Finetuning | ✅ Baseline | Replication of Hegselmann approach |
| **SFT → SW-DPO** | **Finetuning** | **⭐ Core Contribution** | **Severity-conditioned margins from expert annotations** |
| ORPO | Finetuning | 📋 Backup | If DPO OOMs |
| KTO | Finetuning | 📋 Fallback | If preference pairs noisy |
| Self-Refine | Prompt | ❌ Skip | CoVe covers this |
| CAD | Decoding | ❌ Skip | Too complex, 2× inference |
| NER Grounding | Input | ❌ Skip | Entity DB quality concern |
| GRPO | Finetuning | ❌ Skip | Unproven for summarization |
| SPIN | Finetuning | ❌ Skip | Not hallucination-specific |

## Novel Thesis Contributions

> **Contribution 1 — Severity-Weighted DPO (SW-DPO)**
>
> We propose using expert-annotated hallucination categories (Hegselmann et al., CHIL 2024) as a
> training signal rather than just an analysis tool. By mapping the 11 hallucination categories to
> clinical severity weights derived from the NCC MERP patient safety index, we create per-sample
> margins in the DPO loss function. This makes the model learn harder from clinically dangerous
> errors (medication fabrication, contradicted facts) while down-weighting benign word-level noise.
>
> **Contribution 2 — DPO + CoVe Stacking**
>
> We test whether combining finetuning-based alignment (SW-DPO) with inference-time verification
> (CoVe) yields synergistic hallucination reduction. The hypothesis: SW-DPO reduces the model's
> tendency to produce high-severity hallucinations (internal change), while CoVe catches remaining
> hallucinations (external check). Together they should outperform either alone.
