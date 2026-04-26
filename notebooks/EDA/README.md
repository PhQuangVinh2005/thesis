# EDA Notebooks

Exploratory Data Analysis for thesis datasets.

## Notebooks

| # | File | Dataset | Status |
|---|------|---------|--------|
| 1 | `01_mimic_iv_ext_bhc_iv_eda.ipynb` | MIMIC-IV-BHC (270K clinical note ↔ BHC summary pairs) | ✅ Done |
| 2 | `02_cross_tokenizer_eda.py` | Cross-tokenizer comparison: GPT-4 vs BioMistral vs Qwen3.5 | ✅ Done |

## Key EDA Findings (MIMIC-IV-BHC)

- **270,033 records**, 5 cols: `note_id`, `input`, `target`, `input_tokens`, `target_tokens`
- **Input**: mean 2,267 ± 914 tokens (GPT-4 tokenizer). **Target**: mean 564 ± 410 tokens
- **Compression ratio**: ~5.8x
- Sections standardized as `<SECTION_NAME>` tags (e.g. `<CHIEF COMPLAINT>`, `<HISTORY OF PRESENT ILLNESS>`)
- De-id token: `___` (replaces names, dates, addresses) — useful as hallucination trap
- **Distribution by range**: 5.4% at 0-1K, 39% at 1K-2K, 52.1% at 2K-4K, 3.5% at 4K+
- **~30% hallucinations** and **~35% omissions** in doctor-written summaries (per paper's Fig 5b)

## Dataset Notes

Detailed notes for each dataset are in `data/raw/`:
- `01_mimic_iv_ext_bhc_iv_note.md` — MIMIC-IV-BHC analysis & thesis mapping
