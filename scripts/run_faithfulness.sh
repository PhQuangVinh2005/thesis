#!/bin/bash
# Faithfulness evaluation — runs SummaC → AlignScore → merge.
# Usage:
#   bash scripts/run_faithfulness.sh outputs/baseline/qwen3_5_2b/range_0_1k/predictions.jsonl
#   bash scripts/run_faithfulness.sh outputs/baseline/qwen3_5_2b/
#   bash scripts/run_faithfulness.sh outputs/baseline/qwen3_5_2b/ --max-samples 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT="$1"
shift
EXTRA_ARGS="$@"

eval "$(conda shell.bash hook)"

# Find prediction files
PRED_FILES=()
if [[ -f "$INPUT" ]]; then
    PRED_FILES=("$INPUT")
elif [[ -d "$INPUT" ]]; then
    while IFS= read -r f; do
        PRED_FILES+=("$f")
    done < <(find "$INPUT" -name "predictions.jsonl" -path "*/range_*/*" | sort)
fi

if [[ ${#PRED_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No predictions.jsonl found in $INPUT"
    exit 1
fi

echo "Faithfulness eval: ${#PRED_FILES[@]} files"

for PRED in "${PRED_FILES[@]}"; do
    PRED_DIR="$(dirname "$PRED")"
    echo ""
    echo "=== $PRED ==="

    # SummaC
    conda activate eval_summac
    python "$PROJECT_ROOT/scripts/run_evaluation.py" \
        --predictions "$PRED" \
        --phase faithfulness \
        --metrics summac \
        --scores-file summac_scores.jsonl \
        --summary-file summac_summary.json \
        $EXTRA_ARGS
    conda deactivate

    # AlignScore
    conda activate eval_align
    python "$PROJECT_ROOT/scripts/run_evaluation.py" \
        --predictions "$PRED" \
        --phase faithfulness \
        --metrics alignscore \
        --scores-file align_scores.jsonl \
        --summary-file align_summary.json \
        $EXTRA_ARGS
    conda deactivate

    # Merge
    python "$PROJECT_ROOT/scripts/merge_faith_scores.py" "$PRED_DIR"
done

echo ""
echo "✓ Done"
