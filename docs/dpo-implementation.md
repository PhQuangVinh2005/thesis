# DPO Finetuning — Implementation Plan

> Finetune Qwen3.5-4B with QLoRA + DPO via Unsloth to reduce hallucination in clinical summarization.
> **Novel contribution**: Severity-Weighted DPO (SW-DPO) — per-sample margins from expert hallucination categories.

## Hardware & Environment

| Component | Spec |
|-----------|------|
| GPU | RTX 5060 Ti 16GB VRAM |
| RAM | 32GB |
| CPU | AMD R7 5700X |
| OS | Ubuntu Server 26.04 LTS |
| CUDA | 13.0 |
| Driver | 580.142 |
| Framework | Unsloth (confirmed Qwen3.5 support) |

## Target Model

- **Model**: Qwen3.5-4B
- **Quantization**: QLoRA (4-bit NF4)
- **VRAM estimate**: ~8-11 GB for SFT, ~10-12 GB for DPO (with reference model)
- **Architecture note**: Qwen3.5 uses hybrid attention (Gated DeltaNet + full attention). Requires `flash-linear-attention` library for DeltaNet layers. Standard `flash_attention_2` should NOT be applied globally — only to the full-attention layers.

## Key Decision: Why DPO?

| Method | Evidence | VRAM | Why chosen/rejected |
|--------|----------|------|---------------------|
| **DPO** | 40% medical hallucination reduction (Tian et al., ICLR 2024) | ~10-12 GB | Proven for medical factuality, simple classification loss, fits 16GB |
| **SW-DPO** ⭐ | Factuality-conditioned margins (Chaduvula et al., arXiv:2601.03027) | ~10-12 GB | Novel: uses expert severity labels in DPO margin |
| PPO/RLHF | Good but unstable | >16 GB | Needs separate reward model — won't fit 16GB |
| ORPO | Competitive, simpler | ~6-8 GB | Backup — no proven factuality results yet |
| KTO | Works with binary labels | ~7-9 GB | Fallback — if preference pairs are too noisy |
| GRPO | Math/reasoning focused | ~8-12 GB | Unproven for summarization tasks |

**Papers**:
- DPO: Rafailov et al., NeurIPS 2023, arXiv:2305.18290
- FactTune: Tian et al., "Fine-tuning Language Models for Factuality", ICLR 2024, arXiv:2311.08401
- F-DPO: Chaduvula et al., "Reducing Hallucinations via Factuality-Aware Preference Learning", arXiv:2601.03027
- NCC MERP: "Index for Categorizing Medication Errors", 2001 (severity taxonomy)

---

## Dataset Strategy

### Key Decision: No public clinical DPO dataset exists

Existing datasets evaluated:

| Dataset | Why it doesn't fit |
|---------|-------------------|
| UltraFeedback (250K pairs) | Not medical domain, not faithfulness-specific |
| Hallucinations-MIMIC-DI (200 samples) | Binary labels only, too small, different task |
| ACI-BENCH annotations (~200 samples) | Too small, different domain (dialogue→SOAP) |
| MedHal (2025) | Binary labels, different tasks |

### Decision: Self-generate preference pairs from existing baseline predictions

**Rationale**: Tian et al. (ICLR 2024) proved that self-generated preference pairs using automated scorers **outperform** human-labeled RLHF. This approach is methodologically sound and has published precedent.

**Academic citation for paper**:
> "Following the methodology of Tian et al. (2024), we construct preference pairs for DPO training using automated faithfulness metrics (AlignScore, SummaC) rather than human annotations."

### Why LM outputs instead of ground truth summaries?

DPO needs **pairs** (chosen + rejected) for the same input. The ground truth is only ONE summary — you can't pair it.

| Approach | Problem |
|----------|---------|
| Ground truth as "chosen" + LM output as "rejected" | Model learns to mimic human STYLE instead of learning FAITHFULNESS. The human writing style differs from LM style — DPO isolates style, not factuality. |
| **Best LM output as "chosen" + Worst LM output as "rejected"** ✅ | Both are in the same distribution. The ONLY difference is faithfulness score. DPO isolates the factuality signal. |

**Ground truth IS used** — but in SFT (Phase D), not DPO (Phase E). Each stage uses the right data type for its purpose.

### Why is non-human-annotated data acceptable?

| Factor | Human Annotators | Automated Pipeline |
|--------|-----------------|-------------------|
| Consistency | Varies, fatigue effects | Perfectly consistent |
| Scale | Expensive, slow (~200 labels) | Free, fast (7,500 scored) |
| Medical expertise | Need clinical experts ($$$) | AlignScore/SummaC trained on NLI |
| Reproducibility | Hard to reproduce | 100% reproducible |

### How many samples is enough?

| Scale | Samples | Context |
|-------|---------|---------|
| Minimum | 500-1,000 | Measurable improvement |
| **Recommended** | **1,500-3,000** | **Sweet spot for domain-specific** |
| Production | 5,000-10,000+ | Overkill for thesis |

**Our approach**: Quality > quantity. Instead of 1,464 auto-generated pairs (metric-based), we use
**100 expert-annotated golden pairs** from Hegselmann et al. (CHIL 2024). Doctor-annotated
hallucinations provide stronger, cleaner signal than automated faithfulness metrics.

**Novel extension**: Each golden pair includes **11-category hallucination span annotations**
(see `hallucinations_mimic_di.jsonl`). We use these to compute **per-sample severity scores**
for the SW-DPO margin (see Phase E2).

### Hallucination Category Distribution (from expert annotations)

| Category | Spans | % | Clinical Risk | Severity Weight |
|----------|-------|---|---------------|----------------|
| `word_unsupported` | 76 | 26.6% | Low | 1.0 |
| `condition_unsupported` | 52 | 18.2% | **HIGH** | 3.5 |
| `time_unsupported` | 35 | 12.2% | Medium | 2.0 |
| `medication_unsupported` | 34 | 11.9% | **HIGHEST** | 4.0 |
| `location_unsupported` | 29 | 10.1% | Medium | 2.0 |
| `procedure_unsupported` | 19 | 6.6% | HIGH | 3.0 |
| `name_unsupported` | 18 | 6.3% | Medium | 1.5 |
| `contradicted_fact` | 15 | 5.2% | **HIGHEST** | 5.0 |
| `number_unsupported` | 7 | 2.4% | HIGH | 3.0 |
| `other_unsupported` | 1 | 0.3% | Low | 1.0 |

**Source**: 100 doctor-written summaries, 286 total hallucination spans, avg 3.1 spans/sample.
Severity weights derived from NCC MERP patient safety index (no human annotation needed).

### Dataset construction — Golden Expert Pairs (ACTIVE)

Source: PhysioNet `medical-expert-annotations-of-unsupported-facts-in-doctor-written-and-llm-generated-patient-summaries`

For each of the 100 BHC→patient summary pairs:
1. `prompt` = clinical context (Brief Hospital Course text) with instruction template
2. `chosen` = `cleaned_improved` summary (hallucinations removed by medical experts)
3. `rejected` = `original` summary (doctor-written, with hallucinations intact)

**3-way ablation**: Nested subsets 10 ⊂ 50 ⊂ 100 (seed=42) to study data scale effect.

**Task mismatch acknowledged**: Golden data is BHC→patient summary (patient-facing language),
our pipeline is notes→BHC (clinical language). Accepted because the faithfulness signal
(what constitutes a hallucination in clinical text) transfers across tasks.

**10 held-out validation pairs** for sanity checking during training.

### Dataset construction — Auto-Generated Pairs (ARCHIVED)

For each of the 1,500 `sample_id`s in MIMIC-IV-BHC:

1. All 5 models already generated baseline predictions on the same 1,500 samples ✅
2. All 5 models already have faithfulness scores (SummaC + AlignScore) ✅
3. Compute composite score: `composite = 0.6 × alignscore + 0.4 × summac_conv`
4. `chosen` = prediction from model with HIGHEST composite score
5. `rejected` = prediction from model with LOWEST composite score
6. Filter: discard pairs where `|chosen_score - rejected_score| < 0.05`

Output: `data/processed/dpo/preference_pairs.jsonl` (1,464 pairs). Kept for potential comparison.

### Model roles in dataset generation (actual results)

| Model | Chosen | Rejected | Primary role |
|-------|--------|----------|-------------|
| BioMistral-7B | 215 (14.7%) | 685 (46.8%) | 🔴 Rejected source |
| BioMistral-7B-SLERP | 349 (23.8%) | 288 (19.7%) | 🟠 Mixed |
| Qwen3.5-2B | 378 (25.8%) | 129 (8.8%) | 🟢 Chosen source |
| Qwen3.5-4B | 240 (16.4%) | 190 (13.0%) | 🟡 Mixed (on-policy) |
| Qwen3.5-9B | 282 (19.3%) | 172 (11.7%) | 🟢 Chosen source |

---

## Pipeline: 6 Phases

### Phase A: Data (DONE ✅)

Already completed:
- 5 models × 3 ranges × 500 samples = 7,500 predictions
- Full 500-sample faithfulness scores (SummaC + AlignScore) for all 15 baseline runs
- Predictions: `outputs/baseline/{model}/range_{range}/`
- Faith scores: `data/raw/finetune/dpo/baseline/{model}/range_{range}/`

### Phase B: Build DPO Preference Pairs (DONE ✅)

**Script**: `scripts/build_dpo_dataset.py`

**Input**: `predictions.jsonl` from `outputs/baseline/` + `faith_scores.jsonl` from `data/raw/finetune/dpo/baseline/`

**Output**: `data/processed/dpo/preference_pairs.jsonl` (1,464 pairs) + `dataset_stats.json`

**Filters applied**:
- Score gap ≥ 0.05 (removed 22 pairs)
- Degenerate copy-paste outputs filtered (summary/context ratio > 0.5, excluded 446 model predictions, removed 14 pairs)

**Actual results**:
- 1,500 candidates → **1,464 preference pairs**
- Chosen score: 0.633 ± 0.099, Rejected: 0.393 ± 0.105, Gap: 0.240 ± 0.107
- See `docs/composite-score-analysis.md` for weight justification

### Phase C: Qwen3.5-4B HF Migration

Switch from Ollama to HuggingFace Transformers via Unsloth.

```bash
pip install unsloth unsloth_zoo
pip install flash-linear-attention
pip install bitsandbytes accelerate trl peft datasets
```

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)
```

**Verify**: Run 5-10 samples and compare output quality with Ollama baseline.

### Phase D: SFT (Stage 1 — Domain Adaptation)

**Purpose**: Teach clinical format + medical vocabulary. Uses ground truth summaries.

**Data format**:
```json
{
    "messages": [
        {"role": "system", "content": "You are an expert medical professional..."},
        {"role": "user", "content": "<clinical notes>"},
        {"role": "assistant", "content": "<ground truth BHC>"}
    ]
}
```

**Config**:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
)

# TrainingArguments:
#   per_device_train_batch_size=2
#   gradient_accumulation_steps=8  (effective batch=16)
#   num_train_epochs=3
#   learning_rate=2e-4
#   gradient_checkpointing=True
#   bf16=True
#   max_seq_length=2048
```

**Time**: ~3-5 hours on RTX 5060 Ti

### Phase E: DPO-Uniform (Stage 2 — Replication Baseline) — 3-Way Size Ablation

**Purpose**: Align model to prefer faithful over hallucinated summaries. Standard DPO without severity margins.
**This serves as the control for the SW-DPO experiment (Phase E2).**

**3 runs** from the same SFT checkpoint, varying only dataset size:

**Config**:
```python
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,           # Unsloth handles internally
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    args=DPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,       # DPO overfits fast
        learning_rate=5e-6,       # Much lower than SFT
        beta=0.1,                 # DPO temperature
        gradient_checkpointing=True,
        bf16=True,
    ),
    max_length=2048,
    max_prompt_length=1536,
)
```

**Overfitting watch**: Especially critical with 10-pair run. Monitor `reward_accuracy`. Target: 0.65-0.85. Stop if >0.90.

| Run | Pairs | Output adapter | Expected time |
|-----|-------|----------------|---------------|
| DPO-U-10 | 10 golden pairs | `models/qwen35_4b_dpo_uniform_10_lora/` | ~10 min |
| DPO-U-50 | 50 golden pairs | `models/qwen35_4b_dpo_uniform_50_lora/` | ~30 min |
| DPO-U-100 | 100 golden pairs | `models/qwen35_4b_dpo_uniform_100_lora/` | ~1 hr |

### Phase E2: SW-DPO (Stage 2 — Core Contribution) — 3-Way Severity Ablation ⭐

**Purpose**: Test whether severity-conditioned margins improve DPO for clinical hallucination reduction.

**What's different from Phase E**: The DPO loss margin varies per sample based on the clinical
severity of hallucinations in the rejected response. Samples with dangerous errors
(medication fabrication, contradicted facts) get a larger margin than samples with only
benign word-level issues.

**Loss formulation**:
```
Standard DPO:
  L = -log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

SW-DPO (severity-conditioned margin):
  severity(y_l) = Σ weight(category_i) × count(category_i) for all spans in rejected
  norm_sev = severity(y_l) / max_severity_in_dataset

  L = -log σ(β · (log-ratio_chosen - log-ratio_rejected) - α · norm_sev)

where:
  α ∈ {0.5, 1.0, 2.0}  — margin strength hyperparameter
  norm_sev ∈ [0, 1]     — min-max normalized across dataset
```

**Severity weight table** (from NCC MERP patient safety framework):
```python
SEVERITY_WEIGHTS = {
    "contradicted_fact":      5.0,  # Direct contradiction → wrong treatment risk
    "medication_unsupported":  4.0,  # Wrong drug/dose → adverse events
    "condition_unsupported":   3.5,  # Wrong diagnosis → wrong treatment plan
    "procedure_unsupported":   3.0,  # Wrong procedure → misleading follow-up
    "number_unsupported":      3.0,  # Wrong lab value → clinical decisions
    "time_unsupported":        2.0,  # Wrong timing → scheduling errors
    "location_unsupported":    2.0,  # Wrong body part → usually obvious
    "name_unsupported":        1.5,  # Wrong specialist → low clinical impact
    "word_unsupported":        1.0,  # Stylistic/word-level → minimal risk
    "other_unsupported":       1.0,  # Catch-all → minimal impact
}
```

**Implementation**: Subclass TRL `DPOTrainer` to inject per-sample margins:
```python
class SeverityWeightedDPOTrainer(DPOTrainer):
    """DPO with per-sample severity-conditioned margins."""

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        metrics = {}
        # Standard DPO forward pass
        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(model, batch)
        ref_chosen_logps, ref_rejected_logps = ...  # from reference model

        # Compute log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # SW-DPO: inject per-sample severity margin
        severity_margins = batch["severity_margin"]  # from dataset
        logits = self.beta * (chosen_logratios - rejected_logratios) - severity_margins

        losses = -F.logsigmoid(logits)
        return losses.mean(), metrics
```

**3-way severity ablation** (all using 100 golden pairs):

| Variant | Margin formula | What it tests |
|---------|---------------|---------------|
| **DPO-Uniform** | β fixed, no margin (α=0) | Standard replication (from Phase E) |
| **SW-DPO-Binary** | margin = α × {0 if clean, 1 if any hallucination} | Does *any* severity signal help? |
| **SW-DPO-Severity** | margin = α × normalized_severity_score | Does *graded* severity help? |

| Run | Output adapter | Description |
|-----|----------------|-------------|
| SW-DPO-Binary-100 | `models/qwen35_4b_swdpo_binary_100_lora/` | Binary severity signal |
| SW-DPO-Sev-100 | `models/qwen35_4b_swdpo_severity_100_lora/` | Full severity weighting |

**Expected finding**: SW-DPO-Severity should outperform both on medication/condition error
reduction, while DPO-Uniform may over-penalize harmless `word_unsupported` errors.

### Phase F: Evaluation & Combination

1. Run SFT model inference → evaluate completeness + faithfulness
2. Run DPO-Uniform (10/50/100) inference → evaluate
3. Run SW-DPO-Binary + SW-DPO-Severity inference → evaluate
4. Sanity check: all models on 10 golden validation samples
5. Run best model through CoVe pipeline (finetuning + inference-time combined)
6. **Category-level analysis**: breakdown errors by hallucination type (NEW)
7. Build final comparison table

**Final comparison**:
```
                                  Faithfulness    Completeness    High-Severity↓
Baseline (Qwen 4B)                    X.XX           X.XX            —
Few-Shot 10                           X.XX           X.XX            —
CoVe                                  X.XX           X.XX            —
SFT only                              X.XX           X.XX            —
DPO-Uniform-100 (replication)         X.XX           X.XX            X.XX
SW-DPO-Binary-100                     X.XX           X.XX            X.XX
SW-DPO-Severity-100             ← ⭐ X.XX           X.XX            X.XX
SW-DPO-Severity-100 + CoVe      ← ⭐ X.XX           X.XX            X.XX
```

**New metric: High-Severity Hallucination Rate**
Breakdown evaluation by hallucination category (medication, condition, contradicted_fact)
to show that SW-DPO specifically reduces dangerous errors more than uniform DPO.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Unsloth API changes | Pin version in requirements |
| DPO overfits on 10-100 pairs | Early stopping, try beta=0.05-0.5, monitor reward_accuracy closely |
| 10 pairs insufficient signal | Part of the ablation — if DPO-10 fails, DPO-50/100 may still work |
| Task mismatch hurts transfer | Golden data faithfulness signal is domain-general; evaluate on BHC task |
| 4B model OOM during DPO | batch_size=1 + grad_accum=32, or switch to ORPO (no ref model) |
| FLA installation fails | Fallback: `attn_implementation="eager"` |
| SW-DPO margin too aggressive | Sweep α ∈ {0.5, 1.0, 2.0}; ablation shows if severity helps or hurts |
| Severity weights subjective | NCC MERP-derived, empirically validated by 3-way ablation |
| TRL DPOTrainer internals change | Pin TRL version; subclass only touches `get_batch_loss_metrics` |
