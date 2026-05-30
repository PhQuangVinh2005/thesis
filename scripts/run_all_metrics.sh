#!/bin/bash
# ==============================================================================
# Run All Evaluation Metrics (Completeness + Faithfulness) for 12 Variants
# ==============================================================================

set -eo pipefail

# Ensure conda env is activated
eval "$(conda shell.bash hook)"
conda activate vinhthesis2

echo "=============================================================================="
echo "🚀 Starting Full 12-Variant Evaluation Metrics Suite"
echo "   Timestamp: $(date)"
echo "=============================================================================="

VARIANTS=(
    "qwen35_4b_base_dpo_10"
    "qwen35_4b_base_dpo_50"
    "qwen35_4b_base_dpo_100"
    "qwen35_4b_base_swdpo_10"
    "qwen35_4b_base_swdpo_50"
    "qwen35_4b_base_swdpo_100"
    
    "qwen35_4b_sft_dpo_10"
    "qwen35_4b_sft_dpo_50"
    "qwen35_4b_sft_dpo_100"
    "qwen35_4b_sft_swdpo_10"
    "qwen35_4b_sft_swdpo_50"
    "qwen35_4b_sft_swdpo_100"
)

TOTAL=${#VARIANTS[@]}
CURRENT=0

for name in "${VARIANTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    exp_dir="outputs/dpo/$name"
    
    echo "------------------------------------------------------------------------------"
    echo "Evaluating [${CURRENT}/${TOTAL}] variant: $name"
    echo "   • Dir: $exp_dir"
    echo "------------------------------------------------------------------------------"
    
    if [ ! -d "$exp_dir" ]; then
        echo "⚠️  Experiment directory '$exp_dir' does not exist. Skipping."
        continue
    fi
    
    # ── Phase 1: Completeness ────────────────────────────────────────────────
    echo "📊 Running Completeness Evaluation..."
    python scripts/run_evaluation.py \
        --experiment-dir "$exp_dir" \
        --phase completeness
        
    # ── Phase 2: Faithfulness ────────────────────────────────────────────────
    echo "📊 Running Faithfulness Evaluation..."
    python scripts/run_evaluation.py \
        --experiment-dir "$exp_dir" \
        --phase faithfulness
        
    echo "✅ Variant $name metrics generated successfully!"
done

echo "=============================================================================="
echo "🎉 ALL 12 ABLATION STUDY VARIANTS EVALUATION COMPLETED!"
echo "   Timestamp: $(date)"
echo "=============================================================================="
