#!/bin/bash
# ==============================================================================
# Run All Ablation Studies for SW-DPO Thesis
# This script executes all 12 runs in the ablation study sequence sequentially.
# It uses the vinhthesis2 conda environment and handles resuming/logging cleanly.
# ==============================================================================

set -eo pipefail

# Ensure conda env is activated
eval "$(conda shell.bash hook)"
conda activate vinhthesis2

echo "=============================================================================="
echo "🚀 Starting Full 12-Variant Ablation Study Suite"
echo "   Environment: vinhthesis2"
echo "   Timestamp: $(date)"
echo "=============================================================================="

# Array of all runs to execute
# Format: model | method | pairs | output_dir
declare -a RUNS=(
    # --- BASE MODEL RUNS ---
    "qwen35_4b dpo 10 models/qwen35_4b_base_dpo_10_lora"
    "qwen35_4b dpo 50 models/qwen35_4b_base_dpo_50_lora"
    "qwen35_4b dpo 100 models/qwen35_4b_base_dpo_100_lora"
    "qwen35_4b swdpo 10 models/qwen35_4b_base_swdpo_10_lora"
    "qwen35_4b swdpo 50 models/qwen35_4b_base_swdpo_50_lora"
    "qwen35_4b swdpo 100 models/qwen35_4b_base_swdpo_100_lora"
    
    # --- SFT MODEL RUNS ---
    "qwen35_4b_sft dpo 10 models/qwen35_4b_sft_dpo_10_lora"
    "qwen35_4b_sft dpo 50 models/qwen35_4b_sft_dpo_50_lora"
    "qwen35_4b_sft dpo 100 models/qwen35_4b_sft_dpo_100_lora"
    "qwen35_4b_sft swdpo 10 models/qwen35_4b_sft_swdpo_10_lora"
    "qwen35_4b_sft swdpo 50 models/qwen35_4b_sft_swdpo_50_lora"
    "qwen35_4b_sft swdpo 100 models/qwen35_4b_sft_swdpo_100_lora"
)

TOTAL_RUNS=${#RUNS[@]}
CURRENT_RUN=0

for run in "${RUNS[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    # Unpack run configurations
    read -r model method pairs output_dir <<< "$run"
    
    echo "------------------------------------------------------------------------------"
    echo "⏳ [Run ${CURRENT_RUN}/${TOTAL_RUNS}] Starting training..."
    echo "   • Base Model: $model"
    echo "   • Method:     $method"
    echo "   • Pairs:      $pairs"
    echo "   • Output Dir: $output_dir"
    echo "------------------------------------------------------------------------------"
    
    # Select script based on method
    if [ "$method" = "dpo" ]; then
        script="scripts/train_dpo.py"
    else
        script="scripts/train_swdpo.py"
    fi
    
    # Check if run output already exists (to avoid duplicate training if restarted)
    if [ -d "$output_dir" ] && [ -f "$output_dir/training_meta.json" ]; then
        echo "⏭️  Output dir '$output_dir' with metadata already exists. Skipping Run ${CURRENT_RUN}."
        continue
    fi
    
    # Clean up output dir if it partially exists
    rm -rf "$output_dir"
    
    # Run the training script
    python "$script" \
        --base-model "$model" \
        --n-pairs "$pairs" \
        --epochs 5 \
        --output-dir "$output_dir"
        
    echo "✅ [Run ${CURRENT_RUN}/${TOTAL_RUNS}] Completed successfully!"
done

echo "=============================================================================="
echo "🎉 ALL 12 ABLATION STUDY RUNS COMPLETED SUCCESSFULLY!"
echo "   Timestamp: $(date)"
echo "=============================================================================="
