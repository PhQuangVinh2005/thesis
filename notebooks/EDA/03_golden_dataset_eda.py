# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Golden Dataset EDA: Hallucination-Free Patient Summaries
#
# **Source**: Hegselmann et al., *"A Data-Centric Approach To Generate Faithful
# and High Quality Patient Summaries with Large Language Models"* (CHIL 2024)
#
# **PhysioNet**: `medical-expert-annotations-of-unsupported-facts-in-doctor-
# written-and-llm-generated-patient-summaries-1.0.1`
#
# ## Purpose
#
# Explore the **cleaned & improved** (golden) dataset — 100 doctor-written
# patient summaries from MIMIC-IV where two medical experts:
#
# 1. Annotated all unsupported facts (hallucinations) using an 11-label protocol
# 2. Manually removed or replaced every hallucination
# 3. Further improved the summaries for quality
#
# **Goal**: Understand what makes these summaries "hallucination-free" to select
# good few-shot examples for the thesis experiments.
#
# ## Dataset Lineage
#
# ```
# MIMIC-IV-Note (discharge notes)
#   └─ MIMIC-IV-Note-Ext-DI (100,175 context-summary pairs)
#       └─ MIMIC-IV-Note-Ext-DI-BHC (same, Brief Hospital Course as context)
#           └─ *_4000_600_chars (26,178 pairs, context ≤4000, summary ≥600)
#               └─ 100 random pairs → annotated by 2 medical experts
#                   ├─ hallucinations_mimic_di.jsonl (with hallucination labels)
#                   ├─ derived/original.json       (raw doctor-written)
#                   ├─ derived/cleaned.json         (hallucinations removed)
#                   └─ derived/cleaned_improved.json (cleaned + quality improved)
#                                                    ← THIS IS THE GOLDEN DATASET
# ```

# %%
import json
from pathlib import Path
from collections import Counter
from textwrap import fill

# ── Paths ──
DATA_ROOT = Path("../../data/raw/medical-expert-annotations-of-unsupported-facts-"
                 "in-doctor-written-and-llm-generated-patient-summaries-1.0.1")

HALLUC_DIR = DATA_ROOT / "hallucination_datasets"
DERIVED_DIR = DATA_ROOT / "derived_datasets"

# ── Load all 3 stages ──
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

golden    = load_jsonl(DERIVED_DIR / "hallucinations_mimic_di_cleaned_improved.json")
cleaned   = load_jsonl(DERIVED_DIR / "hallucinations_mimic_di_cleaned.json")
original  = load_jsonl(DERIVED_DIR / "hallucinations_mimic_di_original.json")
annotated = load_jsonl(HALLUC_DIR / "hallucinations_mimic_di.jsonl")

print(f"Original (doctor-written):   {len(original)} records")
print(f"Cleaned (halluc removed):    {len(cleaned)} records")
print(f"Golden (cleaned+improved):   {len(golden)} records")
print(f"Annotated (with labels):     {len(annotated)} records")

# %% [markdown]
# ## 1. Data Structure
#
# Each record has two fields:
# - **`text`** — the **Brief Hospital Course** (BHC) section from the discharge
#   note. This is the clinical context written by doctors for medical staff.
# - **`summary`** — the **Discharge Instructions** (DI) section, rewritten as a
#   patient-facing summary. This is what patients receive upon leaving.
#
# The annotated version adds a **`labels`** field with span-level hallucination
# annotations.

# %%
# Show the schema
print("=== Golden dataset record keys ===")
print(list(golden[0].keys()))
print()
print("=== Annotated dataset record keys ===")
print(list(annotated[0].keys()))

# %% [markdown]
# ## 2. Full Example: Context → Golden Summary
#
# Let's look at a complete pair to understand the relationship between the BHC
# (clinical notes for doctors) and the DI (patient-facing summary).

# %%
def show_pair(records, idx, title=""):
    """Display a context-summary pair with clear formatting."""
    r = records[idx]
    text = r["text"]
    summary = r["summary"]
    print(f"{'='*70}")
    print(f"  {title} — Record #{idx}")
    print(f"  Context length: {len(text):,} chars | Summary length: {len(summary):,} chars")
    print(f"{'='*70}")
    print()
    print("── CONTEXT (Brief Hospital Course) ──")
    print()
    print(fill(text, width=80))
    print()
    print("── SUMMARY (Discharge Instructions — Golden) ──")
    print()
    print(fill(summary, width=80))
    print()

show_pair(golden, 0, "Sample A: Short case (double vision)")

# %%
show_pair(golden, 50, "Sample B: Complex case (unresponsiveness + fall)")

# %%
show_pair(golden, 99, "Sample C: Multi-problem case (nausea, aortic dissection)")

# %% [markdown]
# ## 3. How Hallucinations Were Removed: Original → Cleaned → Improved
#
# The paper defines 11 hallucination types. Let's see exactly what was changed
# at each stage for a single record. This is key to understanding what makes
# the golden dataset "faithful."

# %%
def show_evolution(idx):
    """Show how a summary evolves from original → cleaned → improved."""
    orig_text = original[idx]["summary"]
    clean_text = cleaned[idx]["summary"]
    gold_text = golden[idx]["summary"]

    # Get hallucination annotations for this record
    anno = annotated[idx]
    labels = anno.get("labels", [])

    print(f"{'='*70}")
    print(f"  Record #{idx} — Evolution through 3 stages")
    print(f"  Context: {len(anno['text']):,} chars | Hallucinations found: {len(labels)}")
    print(f"{'='*70}")
    print()

    if labels:
        print("── HALLUCINATIONS FOUND ──")
        for i, lab in enumerate(labels, 1):
            print(f"  {i}. [{lab['label']}] \"{lab['text']}\"")
            print(f"     Position: chars {lab['start']}–{lab['end']} ({lab['length']} chars)")
        print()

    print(f"── ORIGINAL (doctor-written, {len(orig_text)} chars) ──")
    print(fill(orig_text, width=80))
    print()

    print(f"── CLEANED (hallucinations removed, {len(clean_text)} chars) ──")
    print(fill(clean_text, width=80))
    print()

    print(f"── GOLDEN (cleaned + improved, {len(gold_text)} chars) ──")
    print(fill(gold_text, width=80))
    print()

    # Highlight differences
    if orig_text != clean_text:
        diff_chars = len(orig_text) - len(clean_text)
        print(f"  📝 Cleaning removed {diff_chars} chars ({diff_chars/len(orig_text)*100:.1f}%)")
    if clean_text != gold_text:
        diff_chars = len(clean_text) - len(gold_text)
        print(f"  ✨ Improvement changed {abs(diff_chars)} chars "
              f"({'removed' if diff_chars > 0 else 'added'})")
    if orig_text == clean_text == gold_text:
        print("  ✅ No changes — original was already hallucination-free!")

# Record 0 has 3 hallucinations — good example of cleaning
show_evolution(0)

# %%
# Find a record with many hallucinations to show dramatic cleaning
most_halluc_idx = max(range(len(annotated)),
                      key=lambda i: len(annotated[i].get("labels", [])))
print(f"Record with most hallucinations: #{most_halluc_idx} "
      f"({len(annotated[most_halluc_idx]['labels'])} annotations)")
print()
show_evolution(most_halluc_idx)

# %%
# Show a record with 0 hallucinations (originally clean)
zero_halluc_indices = [i for i, r in enumerate(annotated)
                       if len(r.get("labels", [])) == 0]
print(f"Records with 0 hallucinations: {len(zero_halluc_indices)}")
print(f"Indices: {zero_halluc_indices}")
print()
show_evolution(zero_halluc_indices[0])

# %% [markdown]
# ## 4. The 11 Hallucination Label Types
#
# The paper's annotation protocol defines 11 categories of unsupported facts.
# Understanding these is critical for designing few-shot prompts that teach
# models what to avoid.

# %%
# Collect all labels from annotated dataset
all_labels = []
for r in annotated:
    for lab in r.get("labels", []):
        all_labels.append(lab)

label_counts = Counter(lab["label"] for lab in all_labels)

print(f"Total hallucination annotations: {len(all_labels)} across {len(annotated)} summaries")
print(f"Avg hallucinations per summary: {len(all_labels)/len(annotated):.1f}")
print()
print("Label type distribution:")
print(f"{'Label':<30} {'Count':>5}  Examples")
print("-" * 70)

# Group examples by label for display
label_examples = {}
for lab in all_labels:
    label_examples.setdefault(lab["label"], []).append(lab["text"])

for label, count in label_counts.most_common():
    examples = label_examples[label][:2]
    examples_str = " | ".join(f'"{e[:40]}"' for e in examples)
    print(f"{label:<30} {count:>5}  {examples_str}")

# %% [markdown]
# ### Label Descriptions (from the annotation protocol)
#
# | Label | Description | Clinical Risk |
# |-------|-------------|---------------|
# | `word_unsupported` | A single word that is not in the source | Medium |
# | `condition_unsupported` | Medical condition mentioned without support | **High** |
# | `medication_unsupported` | Medication mentioned without support | **High** |
# | `time_unsupported` | Temporal reference not in the source | Medium |
# | `location_unsupported` | Anatomical/spatial detail unsupported | Medium |
# | `procedure_unsupported` | Medical procedure not mentioned in source | **High** |
# | `name_unsupported` | Name/identifier not in source | Low |
# | `contradicted_fact` | Directly contradicts the source text | **Critical** |
# | `number_unsupported` | Numeric value not supported by source | **High** |
# | `other_unsupported` | Catch-all for other unsupported facts | Variable |
# | `reasoning_unsupported` | Clinical reasoning not in source | Medium |

# %% [markdown]
# ## 5. Content Patterns in Golden Summaries
#
# Let's understand the **structure and tone** of the golden summaries — this is
# what we want the model to learn via few-shot examples.

# %%
print("=== How golden summaries typically begin ===")
print()
for i, r in enumerate(golden[:15]):
    first_sentence = r["summary"].split(".")[0] + "."
    print(f"  [{i:2d}] {first_sentence[:100]}")

# %%
print("=== How golden summaries typically end ===")
print()
for i, r in enumerate(golden[:15]):
    last_sentence = r["summary"].rstrip().rsplit(".", 2)[-2] + "." if "." in r["summary"] else r["summary"][-100:]
    print(f"  [{i:2d}] ...{last_sentence.strip()[:100]}")

# %% [markdown]
# ### Key Observations for Prompt Design
#
# The golden summaries follow consistent patterns:
#
# 1. **Patient-facing language**: "You were admitted for..." (2nd person)
# 2. **Clinical accuracy**: Only facts present in BHC, no inference
# 3. **Structure**: Admission reason → Findings → Treatment → Discharge plan
# 4. **De-identified tokens**: `___` preserved (names, dates, etc.)
# 5. **Professional but accessible**: Medical terms are used but explained

# %% [markdown]
# ## 6. Relationship Between Context and Summary
#
# Understanding how the BHC context maps to the DI summary helps us design
# prompts that instruct models on the expected transformation.

# %%
# Show the text-to-summary relationship
print("=== Context → Summary Mapping Patterns ===\n")

for idx in [0, 10, 30, 50, 75]:
    r = golden[idx]
    text = r["text"]
    summary = r["summary"]

    # Check if summary starts with "You were"
    starts_with_you = summary.strip().startswith("You")
    # Check for de-id tokens
    has_deident = "___" in summary
    # Check for section headers in context
    has_sections = any(tag in text for tag in ["#", "ACUTE", "CHRONIC", "ASSESSMENT"])

    print(f"Record #{idx}:")
    print(f"  Context: {len(text):,} chars | Summary: {len(summary):,} chars "
          f"| Ratio: {len(summary)/len(text):.2f}")
    print(f"  Starts with 'You': {starts_with_you} | Has ___: {has_deident} "
          f"| Context has sections: {has_sections}")
    print(f"  Context begins: \"{text[:80]}...\"")
    print(f"  Summary begins: \"{summary[:80]}...\"")
    print()

# %% [markdown]
# ## 7. Candidate Selection for Few-Shot Prompting
#
# For few-shot prompting, we want examples that are:
# - **Representative**: typical BHC → DI transformation
# - **Clean**: no remaining artifacts or ambiguity
# - **Reasonable length**: short enough to fit in prompt budget
# - **Diverse**: cover different medical conditions/complexities
#
# Let's identify candidates.

# %%
# Score each record for few-shot suitability
candidates = []
for i, r in enumerate(golden):
    text = r["text"]
    summary = r["summary"]

    # Get original hallucination count
    orig_halluc_count = len(annotated[i].get("labels", [])) if i < len(annotated) else 0

    record_info = {
        "idx": i,
        "text_len": len(text),
        "summary_len": len(summary),
        "ratio": len(summary) / len(text) if len(text) > 0 else 0,
        "starts_with_you": summary.strip().startswith("You"),
        "orig_halluc_count": orig_halluc_count,
        "context_first_line": text.split("\n")[0][:100],
        "summary_first_sentence": summary.split(".")[0][:100],
    }
    candidates.append(record_info)

# Sort by combined length (text + summary) — shorter = better for few-shot
candidates_sorted = sorted(candidates, key=lambda x: x["text_len"] + x["summary_len"])

print("=== Top 15 shortest pairs (best for few-shot prompt budget) ===\n")
print(f"{'Idx':>4} {'Text':>6} {'Summ':>6} {'Ratio':>6} {'Orig H':>7} First line of context")
print("-" * 90)
for c in candidates_sorted[:15]:
    print(f"{c['idx']:4d} {c['text_len']:6d} {c['summary_len']:6d} "
          f"{c['ratio']:6.2f} {c['orig_halluc_count']:7d}  "
          f"{c['context_first_line'][:50]}")

# %%
# Show the top 5 shortest complete pairs
print("=== Top 5 Shortest Golden Pairs (Full Content) ===\n")
for rank, c in enumerate(candidates_sorted[:5], 1):
    idx = c["idx"]
    r = golden[idx]
    print(f"{'='*70}")
    print(f"  RANK #{rank} — Record #{idx}")
    print(f"  Context: {c['text_len']} chars | Summary: {c['summary_len']} chars")
    print(f"  Originally had {c['orig_halluc_count']} hallucination(s)")
    print(f"{'='*70}")
    print()
    print("── CONTEXT ──")
    print(fill(r["text"], width=80))
    print()
    print("── GOLDEN SUMMARY ──")
    print(fill(r["summary"], width=80))
    print()

# %% [markdown]
# ## 8. Full Catalog: All 100 Golden Records at a Glance
#
# Quick reference showing the medical topic and dimensions for each record.

# %%
print(f"{'Idx':>3}  {'Text':>5} {'Summ':>5} {'H':>2}  First line of BHC context")
print("-" * 95)
for i, r in enumerate(golden):
    first_line = r["text"].split("\n")[0].replace("Brief Hospital Course: ", "")[:65]
    h_count = len(annotated[i].get("labels", [])) if i < len(annotated) else "?"
    print(f"{i:3d}  {len(r['text']):5d} {len(r['summary']):5d} {h_count:>2}  {first_line}")

# %% [markdown]
# ## 9. Summary of Findings
#
# ### What is the Golden Dataset?
# - **100 doctor-written Discharge Instructions** from MIMIC-IV
# - Each paired with its **Brief Hospital Course** as source context
# - All hallucinations **manually removed** by two medical experts
# - Further **improved** for quality and readability
#
# ### Why it matters for the thesis
# - These are **verified hallucination-free** summaries
# - They demonstrate the **exact transformation** (BHC → DI) we want models to
#   learn
# - They can serve as **few-shot examples** in the prompt to teach models:
#   1. Use patient-facing language ("You were admitted for...")
#   2. Only include facts from the source BHC
#   3. Follow the structure: admission → findings → treatment → discharge
#   4. Preserve de-identified tokens (`___`)
#
# ### Few-shot candidate criteria
# - **Short pairs** (< 2000 chars total) to fit within prompt budget
# - **Had hallucinations originally** (shows the model what "corrected" looks
#   like vs. the original)
# - **Representative medical topics** (not all the same condition)
#
# ### Next steps
# - Select 2–3 few-shot examples from the candidates
# - Design a `BaseTechnique` subclass that injects these into the prompt
# - Compare against baseline (zero-shot) across all models and ranges
