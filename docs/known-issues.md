# Known Issues

## Active Issues

| Issue | Workaround | File |
|-------|-----------|------|
| BERTScore `OverflowError: int too big` | Monkey-patch `model_max_length=512` | `src/evaluation/completeness/bert_score.py` |
| `numpy 2.x` crashes spacy/thinc | Pin `numpy<=1.26.4` | `requirements/main.txt` |
| Evaluation imports fail without deps | Lazy imports via `__getattr__` | `src/evaluation/__init__.py` |
| Factory import fails without torch | Lazy import registry | `src/models/factory.py` |
| bitsandbytes dtype cast warning | Always pass `torch_dtype` | `src/models/hf_model.py` |
| SummaC + AlignScore dep conflict | Separate conda envs | `requirements/summac.txt`, `requirements/align.txt` |
| SummaC-ZS negative scores | Expected: range [-1,+1] | N/A |

## BioMistral + CoVe Incompatibility

**Status**: Won't fix — architectural limitation, not a bug.

BioMistral-7B and BioMistral-7B-SLERP are **excluded from CoVe experiments** because they lack the instruction-following capability required for multi-step structured reasoning.

### Observed Failures

| Symptom | Example |
|---------|---------|
| Token dump drafts | `"M, S, NKA, , , , , , , , ..."` (comma-separated fragments) |
| Trivial questions | Only 1 question generated instead of 5; questions quote source text verbatim |
| No verification structure | Output lacks CONFIRMED/CONTRADICTED/UNVERIFIABLE format |
| No PART 1/PART 2 | Model writes a single summary without the structured two-part response |
| Hallucinations amplified | CoVe produces fabricated procedures (e.g., "laparoscopic sigmoid colectomy") not in source |

### Root Cause

BioMistral is a domain-specific model fine-tuned from Mistral-7B for biomedical text generation. It was not instruction-tuned for multi-step structured reasoning tasks. The CoVe pipeline requires:

1. Following a structured prompt with numbered output format
2. Self-critique via CONFIRMED/CONTRADICTED labels
3. Generating a separate corrected summary section

Qwen3.5 (instruction-tuned) handles all three; BioMistral does not.

### Thesis Implication

> "CoVe requires sufficient instruction-following capability. Domain-specific models fine-tuned without instruction tuning (BioMistral) fail at multi-step structured reasoning, producing degenerate outputs. General instruction-tuned models (Qwen3.5) are better candidates for CoVe."

## BioMistral HF Loading Warnings

**Status**: Non-critical, expected behavior.

When loading BioMistral via HuggingFace Transformers, several warnings appear:

| Warning | Meaning |
|---------|---------|
| `Flash Attention failed... Falling back to eager` | FA2 is incompatible with Mistral architecture; eager attention used instead |
| `httpx.ConnectError` / `OSError: could not create safetensors conversion PR` | Background HF auto-conversion thread fails silently; model loads from `pytorch_model.bin` |

These are handled by the fallback logic in `src/models/hf_model.py` and do not affect inference quality.
