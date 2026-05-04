#!/bin/bash
# Faithfulness evaluation for DPO dataset — ALL 500 samples per model/range.
# Outputs to data/raw/finetune/dpo/baseline/{model}/range_{range}/
# Does NOT touch existing outputs/ scores.
#
# Usage:
#   bash scripts/run_faith_dpo.sh                    # all 5 models
#   bash scripts/run_faith_dpo.sh qwen3_5_4b         # single model
#   bash scripts/run_faith_dpo.sh qwen3_5_2b qwen3_5_9b  # multiple models
#
# Time estimate: ~30-60 min per model (3 ranges × 500 samples × 2 metrics)
# Total: ~3-5 hours for all 5 models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DPO_BASE="$PROJECT_ROOT/data/raw/finetune/dpo/baseline"

ALL_MODELS=(biomistral7b biomistral7b_slerp qwen3_5_2b qwen3_5_4b qwen3_5_9b)

if [[ $# -ge 1 ]]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Faithfulness eval → DPO dataset (500 samples/run)"
echo "  Models: ${MODELS[*]}"
echo "  Output: $DPO_BASE/{model}/range_*/"
echo "═══════════════════════════════════════════════════════════"

eval "$(conda shell.bash hook)"

for model in "${MODELS[@]}"; do
    PRED_DIR="$PROJECT_ROOT/outputs/baseline/$model"
    OUT_DIR="$DPO_BASE/$model"

    if [[ ! -d "$PRED_DIR" ]]; then
        echo "SKIP: $PRED_DIR not found"
        continue
    fi

    echo ""
    echo "▶ Model: $model"
    echo "  Input:  $PRED_DIR"
    echo "  Output: $OUT_DIR"

    for range_dir in "$PRED_DIR"/range_*; do
        range_name="$(basename "$range_dir")"
        PRED_FILE="$range_dir/predictions.jsonl"
        RANGE_OUT="$OUT_DIR/$range_name"

        if [[ ! -f "$PRED_FILE" ]]; then
            echo "  SKIP: $PRED_FILE not found"
            continue
        fi

        mkdir -p "$RANGE_OUT"
        echo ""
        echo "  --- $model/$range_name ---"

        # SummaC
        echo "  [1/3] SummaC..."
        conda activate eval_summac
        python "$PROJECT_ROOT/scripts/run_evaluation.py" \
            --predictions "$PRED_FILE" \
            --phase faithfulness \
            --metrics summac \
            --scores-file summac_scores.jsonl \
            --summary-file summac_summary.json \
            --output-dir "$RANGE_OUT"
        conda deactivate

        # AlignScore
        echo "  [2/3] AlignScore..."
        conda activate eval_align
        python "$PROJECT_ROOT/scripts/run_evaluation.py" \
            --predictions "$PRED_FILE" \
            --phase faithfulness \
            --metrics alignscore \
            --scores-file align_scores.jsonl \
            --summary-file align_summary.json \
            --output-dir "$RANGE_OUT"
        conda deactivate

        # Merge
        echo "  [3/3] Merging..."
        python "$PROJECT_ROOT/scripts/merge_faith_scores.py" "$RANGE_OUT"
    done

    echo "  ✓ $model complete"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✓ All done."
echo ""
echo "  Verify counts:"
for model in "${MODELS[@]}"; do
    for r in range_0_1k range_1k_2k range_2k_4k; do
        f="$DPO_BASE/$model/$r/faith_scores.jsonl"
        if [[ -f "$f" ]]; then
            echo "    $model/$r: $(wc -l < "$f") samples"
        else
            echo "    $model/$r: MISSING"
        fi
    done
done
echo "═══════════════════════════════════════════════════════════"
