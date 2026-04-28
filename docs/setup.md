# Environment Setup

## Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX 5060 Ti 16GB (Blackwell sm_120) |
| CUDA Driver | 580.x / CUDA 13.0 |
| PyTorch | cu130 |
| Python | 3.11 via conda |

## 3 Conda Environments

SummaC and AlignScore have irreconcilable dependency conflicts, so each gets its own env.

| Env | Purpose | Key Deps |
|-----|---------|----------|
| `vinhthesis` | Generation + completeness eval | transformers, flash_attn_3, rouge-score, bert-score |
| `eval_summac` | Faithfulness: SummaC | transformers==4.30.0, huggingface-hub==0.17.0 |
| `eval_align` | Faithfulness: AlignScore | transformers==4.40.0, pytorch-lightning==1.9.5 |

## Automated Setup

```bash
bash scripts/setup_env.sh
```

## Manual Setup

### 1. Main env (`vinhthesis`)

```bash
conda create -n vinhthesis python=3.11 -y
conda activate vinhthesis
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2110
uv pip install -r requirements/main.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

### 2. SummaC env (`eval_summac`)

```bash
conda create -n eval_summac python=3.11 -y
conda activate eval_summac
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements/summac.txt
pip install --no-deps summac
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. AlignScore env (`eval_align`)

```bash
conda create -n eval_align python=3.11 -y
conda activate eval_align
pip install torch --index-url https://download.pytorch.org/whl/cu130
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

## Verify

```bash
conda activate vinhthesis
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -m pytest tests/ -v
```

## GPU Memory Guide

| Model | Backend | VRAM | Notes |
|-------|---------|------|-------|
| Qwen3.5-2B | Ollama | ~2.5 GB | Ollama manages VRAM via `keep_alive` |
| Qwen3.5-4B | Ollama | ~5 GB | |
| Qwen3.5-9B | Ollama | ~10 GB | |
| BioMistral-7B | Transformers | ~10 GB | 8-bit quantization via bitsandbytes |
| BioMistral-7B-SLERP | Transformers | ~10 GB | 8-bit quantization via bitsandbytes |

**Important**: Models are loaded once per experiment run. The experiment runner uses `keep_alive=-1` for Ollama models to prevent mid-run unloading. GPU memory is released via `cleanup()` after each experiment completes.

## Experiment Runtime Estimates

| Technique | Model | Samples | Est. Time |
|-----------|-------|---------|-----------|
| Baseline | Qwen3.5-2B | 500×3 | ~2 hours |
| Baseline | BioMistral-7B | 500×3 | ~8 hours |
| Few-Shot (10) | Qwen3.5-9B | 500×3 | ~6 hours |
| CoVe | Qwen3.5-2B | 500×3 | ~6 hours (3× baseline) |
| CoVe | Qwen3.5-9B | 500×3 | ~12 hours (3× baseline) |
