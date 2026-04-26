#!/bin/bash
# Setup all 3 conda environments for thesis project.
# Usage: bash scripts/setup_env.sh

set -e

MAIN_ENV="vinhthesis"
SUMMAC_ENV="eval_summac"
ALIGN_ENV="eval_align"
PYTHON_VERSION="3.11"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"

echo "=== Thesis Environment Setup (CUDA 13.0) ==="

create_env() {
    if ! conda env list | grep -q "^$1 "; then
        conda create -n "$1" python="${PYTHON_VERSION}" -y
    else
        echo "Env '$1' exists."
    fi
}

# ── 1/3: Main ──────────────────────────────────────────────────────────
echo ""
echo "=== [1/3] ${MAIN_ENV} ==="
create_env "${MAIN_ENV}"
eval "$(conda shell.bash hook)"
conda activate "${MAIN_ENV}"

pip install torch torchvision torchaudio --index-url ${PYTORCH_INDEX}
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2110
uv pip install -r requirements/main.txt
uv pip install -e .
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from src.data.schema import EvalSample; print('Package: OK')"

# ── 2/3: SummaC ───────────────────────────────────────────────────────
echo ""
echo "=== [2/3] ${SUMMAC_ENV} ==="
create_env "${SUMMAC_ENV}"
conda activate "${SUMMAC_ENV}"

pip install torch --index-url ${PYTORCH_INDEX}
pip install -r requirements/summac.txt
pip install --no-deps summac
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python -c "from summac.model_summac import SummaCZS; print('SummaC: OK')"

# ── 3/3: AlignScore ───────────────────────────────────────────────────
echo ""
echo "=== [3/3] ${ALIGN_ENV} ==="
create_env "${ALIGN_ENV}"
conda activate "${ALIGN_ENV}"

pip install torch --index-url ${PYTORCH_INDEX}
pip install -r requirements/align.txt
pip install --no-deps "git+https://github.com/yuh-zha/AlignScore.git"
pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python -m spacy download en_core_web_sm

if [ ! -f models/AlignScore-large.ckpt ]; then
    mkdir -p models/
    wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt -P models/
fi
python -c "from alignscore import AlignScore; print('AlignScore: OK')"

# ── Ollama ─────────────────────────────────────────────────────────────
echo ""
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo ""
echo "=== Done ==="
echo "  conda activate ${MAIN_ENV}"
echo "  ollama pull qwen3.5:{2b,4b,9b}"
