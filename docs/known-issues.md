# Known Issues

| Issue | Workaround | File |
|-------|-----------|------|
| BERTScore `OverflowError: int too big` | Monkey-patch `model_max_length=512` | `src/evaluation/completeness/bert_score.py` |
| `numpy 2.x` crashes spacy/thinc | Pin `numpy<=1.26.4` | `requirements/main.txt` |
| Evaluation imports fail without deps | Lazy imports via `__getattr__` | `src/evaluation/__init__.py` |
| Factory import fails without torch | Lazy import registry | `src/models/factory.py` |
| bitsandbytes dtype cast warning | Always pass `torch_dtype` | `src/models/hf_model.py` |
| SummaC + AlignScore dep conflict | Separate conda envs | `requirements/summac.txt`, `requirements/align.txt` |
| SummaC-ZS negative scores | Expected: range [-1,+1] | N/A |
