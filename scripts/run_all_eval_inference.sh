#!/bin/bash
# ==============================================================================
# Merge LoRA and Run vLLM Inference for All 12 Ablation Variants
# Runs the merge step (in vinhthesis2 env) and the vLLM step (in vllm env).
# ==============================================================================

set -eo pipefail

echo "=============================================================================="
echo "🚀 Starting Full 12-Variant Merging & Inference Suite"
echo "   Timestamp: $(date)"
echo "=============================================================================="

# Array of all runs to process
# Format: name | adapter_dir | merged_dir | output_dir
declare -a VARIANTS=(
    "qwen35_4b_base_dpo_10 models/qwen35_4b_base_dpo_10_lora models/qwen35_4b_base_dpo_10_merged outputs/dpo/qwen35_4b_base_dpo_10"
    "qwen35_4b_base_dpo_50 models/qwen35_4b_base_dpo_50_lora models/qwen35_4b_base_dpo_50_merged outputs/dpo/qwen35_4b_base_dpo_50"
    "qwen35_4b_base_dpo_100 models/qwen35_4b_base_dpo_100_lora models/qwen35_4b_base_dpo_100_merged outputs/dpo/qwen35_4b_base_dpo_100"
    "qwen35_4b_base_swdpo_10 models/qwen35_4b_base_swdpo_10_lora models/qwen35_4b_base_swdpo_10_merged outputs/dpo/qwen35_4b_base_swdpo_10"
    "qwen35_4b_base_swdpo_50 models/qwen35_4b_base_swdpo_50_lora models/qwen35_4b_base_swdpo_50_merged outputs/dpo/qwen35_4b_base_swdpo_50"
    "qwen35_4b_base_swdpo_100 models/qwen35_4b_base_swdpo_100_lora models/qwen35_4b_base_swdpo_100_merged outputs/dpo/qwen35_4b_base_swdpo_100"
    
    "qwen35_4b_sft_dpo_10 models/qwen35_4b_sft_dpo_10_lora models/qwen35_4b_sft_dpo_10_merged outputs/dpo/qwen35_4b_sft_dpo_10"
    "qwen35_4b_sft_dpo_50 models/qwen35_4b_sft_dpo_50_lora models/qwen35_4b_sft_dpo_50_merged outputs/dpo/qwen35_4b_sft_dpo_50"
    "qwen35_4b_sft_dpo_100 models/qwen35_4b_sft_dpo_100_lora models/qwen35_4b_sft_dpo_100_merged outputs/dpo/qwen35_4b_sft_dpo_100"
    "qwen35_4b_sft_swdpo_10 models/qwen35_4b_sft_swdpo_10_lora models/qwen35_4b_sft_swdpo_10_merged outputs/dpo/qwen35_4b_sft_swdpo_10"
    "qwen35_4b_sft_swdpo_50 models/qwen35_4b_sft_swdpo_50_lora models/qwen35_4b_sft_swdpo_50_merged outputs/dpo/qwen35_4b_sft_swdpo_50"
    "qwen35_4b_sft_swdpo_100 models/qwen35_4b_sft_swdpo_100_lora models/qwen35_4b_sft_swdpo_100_merged outputs/dpo/qwen35_4b_sft_swdpo_100"
)

TOTAL=${#VARIANTS[@]}
CURRENT=0

for var in "${VARIANTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    read -r name adapter merged output <<< "$var"
    
    echo "------------------------------------------------------------------------------"
    echo "Processing [${CURRENT}/${TOTAL}] variant: $name"
    echo "------------------------------------------------------------------------------"
    
    # ── Step 1: Merge LoRA (needs vinhthesis2 env) ────────────────────────────
    if [ -f "$merged/config.json" ]; then
        echo "⏭️  Merged model '$merged' already exists. Skipping merge."
    else
        echo "⏳ Step 1/2: Merging $adapter -> $merged..."
        eval "$(conda shell.bash hook)"
        conda activate vinhthesis2
        python scripts/merge_lora.py --adapter "$adapter" --output "$merged"
        
        # vLLM requires config fix for Qwen merged configs sometimes
        if [ -f "scripts/fix_merged_config.py" ]; then
            echo "🔧 Fixing merged config..."
            conda activate vllm
            python scripts/fix_merged_config.py --model-dir "$merged" || echo "Warning: config fix skipped/not needed."
        fi
    fi
    
    # ── Step 2: vLLM Inference (needs vllm env) ──────────────────────────────
    # We check if predictions already exist to support resuming!
    if [ -f "$output/range_2k_4k/predictions.jsonl" ]; then
        echo "⏭️  Inference output for '$name' already exists. Skipping vLLM."
    else
        echo "⏳ Step 2/2: Running vLLM inference on 1,500 samples..."
        eval "$(conda shell.bash hook)"
        conda activate vllm
        
        # Enforce Blackwell compatibility environment variables
        export VLLM_USE_FLASHINFER_SAMPLER=0
        export VLLM_DISABLE_FLASHINFER=1
        
        python scripts/run_sft_inference.py \
            --model "$merged" \
            --output-dir "$output"
    fi
    
    echo "✅ Variant $name fully processed!"
done

echo "=============================================================================="
echo "🎉 ALL 12 ABLATION STUDY VARIANTS MERGED & EVALUATION INFERENCES GENERATED!"
echo "   Timestamp: $(date)"
echo "=============================================================================="
