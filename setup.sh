#!/usr/bin/env bash
# =============================================================================
# setup.sh — install everything and download the Human-1 (Moshi) model weights.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Works on a fresh GPU box (Kaggle / Colab / remote H100 / etc).
# Idempotent: re-running skips already-downloaded weights.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=================================================================="
echo " Human-1 FastAPI server — setup"
echo "=================================================================="

# ----------------------------------------------------------------------------
# 1. Read paths from config.json (single source of truth)
# ----------------------------------------------------------------------------
HF_REPO="$(python3 -c 'import json;print(json.load(open("config.json"))["model"]["hf_repo"])')"
WEIGHTS_DIR="$(python3 -c 'import json;print(json.load(open("config.json"))["model"]["weights_dir"])')"
echo "Model repo : $HF_REPO"
echo "Weights dir: $WEIGHTS_DIR"

# ----------------------------------------------------------------------------
# 2. PyTorch (CUDA build). Skip if torch already present (e.g. Kaggle images).
# ----------------------------------------------------------------------------
python3 -m pip install --upgrade pip wheel

if python3 -c "import torch" 2>/dev/null; then
    echo "torch already installed: $(python3 -c 'import torch;print(torch.__version__)')"
else
    echo "Installing torch (CUDA 12.1 build)..."
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# ----------------------------------------------------------------------------
# 3. All other Python deps (fastapi, pyngrok, moshi, torchao, accelerate, ...)
# ----------------------------------------------------------------------------
echo "Installing Python requirements..."
python3 -m pip install -r requirements.txt

# ----------------------------------------------------------------------------
# 4. Download model weights from Hugging Face into $WEIGHTS_DIR
# ----------------------------------------------------------------------------
echo "Downloading weights for $HF_REPO (this is ~31 GB, may take a while)..."
python3 - <<PY
import json
from huggingface_hub import snapshot_download
cfg = json.load(open("config.json"))["model"]
snapshot_download(
    repo_id=cfg["hf_repo"],
    local_dir=cfg["weights_dir"],
    allow_patterns=[
        cfg["moshi_weight"],
        cfg["mimi_weight"],
        cfg["tokenizer"],
        "*.vocab",
        "config.json",
    ],
)
# Pre-fetch the base repo config/defaults so model load needs no network.
try:
    snapshot_download(repo_id=cfg["base_repo"], allow_patterns=["config.json", "*.json"])
except Exception as e:
    print("warn: could not prefetch base repo config:", e)
print("Weights ready in", cfg["weights_dir"])
PY

echo "=================================================================="
echo " Setup complete. Start the server with:  python3 server.py"
echo "=================================================================="
