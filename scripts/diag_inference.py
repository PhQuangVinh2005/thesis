#!/usr/bin/env python3
"""Diagnostic: profile Unsloth SFT inference to find bottleneck.

Loads the SFT adapter, generates ONE sample, and prints timing for each stage.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ADAPTER_PATH = "models/qwen35_4b_sft_lora/"
TEST_DATA = "data/processed/mimic_iv_bhc/range_0_1k.jsonl"

def timed(label):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f"  [{label}] {elapsed:.2f}s")
            self.elapsed = elapsed
    return Timer()

# ── 1. Load model ──
print("=" * 60)
print("STAGE 1: Model loading")
print("=" * 60)

with timed("import unsloth"):
    from unsloth import FastLanguageModel

import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

with timed("load model + adapter"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(PROJECT_ROOT / ADAPTER_PATH),
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )

# Check device
print(f"  Model device: {next(model.parameters()).device}")
print(f"  Model dtype: {next(model.parameters()).dtype}")

with timed("for_inference"):
    FastLanguageModel.for_inference(model)

# Extract text tokenizer
processor = tokenizer
text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# ── 2. Load one test sample ──
print("\n" + "=" * 60)
print("STAGE 2: Data + Tokenization")
print("=" * 60)

with open(PROJECT_ROOT / TEST_DATA) as f:
    sample = json.loads(f.readline())

prompt_text = f"""You are an expert medical professional specializing in clinical documentation.

Summarize the following clinical notes into a concise Brief Hospital Course. Focus only on clinically significant events, diagnoses, and treatments. Ensure the summary is technically accurate and uses professional medical terminology.

Strictly adhere to the provided context. If a piece of information is not explicitly stated in the input text, do not include it. Avoid inferring results or patient outcomes.

input: {sample['input']}
summary:"""

print(f"  Input chars: {len(sample['input']):,}")
print(f"  Prompt chars: {len(prompt_text):,}")

with timed("apply_chat_template"):
    messages = [{"role": "user", "content": prompt_text}]
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

with timed("tokenize"):
    input_ids = text_tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(model.device)

print(f"  Input tokens: {input_ids.shape[1]:,}")

# ── 3. Generate ──
print("\n" + "=" * 60)
print("STAGE 3: Generation")
print("=" * 60)

attention_mask = torch.ones_like(input_ids)

# Warmup
with timed("warmup (5 tokens)"):
    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=text_tokenizer.pad_token_id,
        )

torch.cuda.synchronize()

with timed("generate (max 512 tokens, greedy)"):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=text_tokenizer.pad_token_id,
        )

torch.cuda.synchronize()

new_tokens = output_ids[0][input_ids.shape[1]:]
output_text = text_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(f"  Output tokens: {len(new_tokens):,}")
print(f"  Output chars: {len(output_text):,}")
tokens_per_sec = len(new_tokens) / (time.perf_counter() - time.perf_counter())  # will be recalculated

# ── 4. GPU stats ──
print("\n" + "=" * 60)
print("STAGE 4: GPU stats")
print("=" * 60)
print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"  Current VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Print first 200 chars of output
print(f"\n  Output preview: {output_text[:200]}...")
