# Environment Setup

## Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX 5060 Ti 16GB (Blackwell, sm_120 / CC 12.0) |
| CUDA Driver | 580.142 (supports up to CUDA 13.x) |
| Python | 3.11 via conda |

## 4 Conda Environments

SummaC and AlignScore have irreconcilable dependency conflicts, so each gets its own env.
The `vinhthesis` env was replaced by `vinhthesis2` (May 2026) due to Blackwell CUDA library issues.

| Env | Purpose | PyTorch | CUDA Toolkit | Key Deps |
|-----|---------|---------|-------------|----------|
| `vinhthesis` | SFT training, generation, completeness eval | 2.12.0+cu130 | 13.0 | unsloth, transformers, xformers |
| **`vinhthesis2`** | **DPO/SW-DPO training only** | 2.10.0+cu128 | 12.8 (native via `conda install -c nvidia cuda-toolkit`) | unsloth, trl, peft, bitsandbytes |
| `eval_summac` | Faithfulness: SummaC | cu128 | — | transformers==4.30.0, huggingface-hub==0.17.0 |
| `eval_align` | Faithfulness: AlignScore | cu128 | — | transformers==4.40.0, pytorch-lightning==1.9.5 |
| `vllm` | vLLM batch inference | cu130 | 13.0 | vllm (Python 3.12) |

> **Note**: The original `vinhthesis` env (PyTorch 2.12+cu130) suffered from scattered CUDA 13 libraries that bitsandbytes couldn't locate at process start. `vinhthesis2` installs the CUDA toolkit natively via conda, placing all libraries in `$CONDA_PREFIX/lib/` where the dynamic linker finds them automatically.

## Automated Setup

```bash
bash scripts/setup_env.sh
```

## Manual Setup

### 1. Main env (`vinhthesis`)

```bash
conda create -n vinhthesis python=3.11 -y
conda activate vinhthesis
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements/main.txt
pip install -r requirements/finetune.txt  # Unsloth + TRL + PEFT
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
# Optional: Flash Attention (XFormers is used by default if FA2 build fails)
# pip install flash-attn --no-build-isolation
```

### 2. SummaC env (`eval_summac`)

```bash
conda create -n eval_summac python=3.11 -y
conda activate eval_summac
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/summac.txt
pip install --no-deps summac
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. AlignScore env (`eval_align`)

```bash
conda create -n eval_align python=3.11 -y
conda activate eval_align
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/align.txt
pip install --no-deps "git+https://github.com/yuh-zha/AlignScore.git"
pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python -m spacy download en_core_web_sm
mkdir -p models/
wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt -P models/
```

### 4. Ollama (for Qwen3.5)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
```

### 5. DPO training env (`vinhthesis2`) — DPO/SW-DPO Only

Clean environment with native CUDA toolkit. **Used only for DPO/SW-DPO training** (Phase 4D). The original `vinhthesis` env remains for SFT training, generation, and completeness evaluation.

```bash
conda create -n vinhthesis2 python=3.11 -y
conda activate vinhthesis2
conda install -c nvidia cuda-toolkit -y  # native CUDA → bitsandbytes JIT works
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install unsloth
pip install --no-deps "trl>=0.24.0" "peft>=0.19.0" "accelerate>=1.13.0" "bitsandbytes>=0.49.0"
pip install datasets sentencepiece scipy xformers
```

**Verify:**
```bash
python -m bitsandbytes  # should print SUCCESS
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

## Verify

```bash
conda activate vinhthesis2
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -m pytest tests/ -v
```

## GPU Memory Guide

| Model | Backend | VRAM | Notes |
|-------|---------|------|-------|
| Qwen3.5-2B | Ollama | ~2.5 GB | Ollama manages VRAM via `keep_alive` |
| Qwen3.5-4B | Ollama | ~5 GB | |
| Qwen3.5-4B | **Unsloth** | **~4-5 GB** | **4-bit NF4 quantization** |
| Qwen3.5-4B | Unsloth (SFT training) | ~10-12 GB | QLoRA + gradient checkpointing |
| Qwen3.5-4B | Unsloth (DPO training) | **~8-10 GB** | QLoRA + `precompute_ref_log_probs` + `max_seq_length=2048` |
| Qwen3.5-9B | Ollama | ~10 GB | |
| BioMistral-7B | Transformers | ~10 GB | 8-bit quantization via bitsandbytes |

**Important**: Models are loaded once per experiment run. The experiment runner uses `keep_alive=-1` for Ollama models to prevent mid-run unloading. GPU memory is released via `cleanup()` after each experiment completes.

## Experiment Runtime Estimates

| Technique | Model | Samples | Est. Time |
|-----------|-------|---------|-----------|
| Baseline | Qwen3.5-2B | 500×3 | ~2 hours |
| Baseline | BioMistral-7B | 500×3 | ~8 hours |
| Few-Shot (10) | Qwen3.5-9B | 500×3 | ~6 hours |
| CoVe | Qwen3.5-2B | 500×3 | ~6 hours (3× baseline) |
| CoVe | Qwen3.5-9B | 500×3 | ~12 hours (3× baseline) |
| **SFT Training** | **Qwen3.5-4B (Unsloth)** | **28,500 train** | **~30-50 hours** |
| DPO Training | Qwen3.5-4B (Unsloth) | 100 pairs | ~2-4 hours |
