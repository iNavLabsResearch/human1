#!/usr/bin/env bash
# =============================================================================
# setup.sh — install deps and download weights for whichever model(s) are
# ENABLED in config.json (human1 and/or veena).
#
# Usage:  chmod +x setup.sh && ./setup.sh
# Idempotent: re-running skips already-downloaded weights.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=================================================================="
echo " Human-1 / Veena server — setup"
echo "=================================================================="

# ----------------------------------------------------------------------------
# Read enabled flags from config.json (single source of truth)
# ----------------------------------------------------------------------------
read -r HUMAN1_ON VEENA_ON SVARA_ON <<<"$(python3 - <<'PY'
import json
c = json.load(open("config.json"))
print(int(c.get("human1", {}).get("enabled", False)),
      int(c.get("veena", {}).get("enabled", False)),
      int(c.get("svara", {}).get("enabled", False)))
PY
)"
echo "human1 enabled: $HUMAN1_ON"
echo "veena  enabled: $VEENA_ON"
echo "svara  enabled: $SVARA_ON"

# ----------------------------------------------------------------------------
# PyTorch (CUDA build). Skip if torch already present (Kaggle/Colab images).
# ----------------------------------------------------------------------------
python3 -m pip install --upgrade pip wheel
if python3 -c "import torch" 2>/dev/null; then
    echo "torch already installed: $(python3 -c 'import torch;print(torch.__version__)')"
else
    echo "Installing torch (CUDA 12.1 build)..."
    python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# ----------------------------------------------------------------------------
# Python deps (fastapi, pyngrok always; model deps as needed)
# ----------------------------------------------------------------------------
echo "Installing Python requirements..."
python3 -m pip install -r requirements.txt

# ----------------------------------------------------------------------------
# Download weights for the enabled model(s)
# ----------------------------------------------------------------------------
if [ "$HUMAN1_ON" = "1" ]; then
    echo "Downloading Human-1 weights (~31 GB)..."
    python3 - <<'PY'
import json
from huggingface_hub import snapshot_download
c = json.load(open("config.json"))["human1"]
snapshot_download(
    repo_id=c["hf_repo"],
    local_dir=c["weights_dir"],
    allow_patterns=[c["moshi_weight"], c["mimi_weight"], c["tokenizer"], "*.vocab", "config.json"],
)
try:
    snapshot_download(repo_id=c["base_repo"], allow_patterns=["*.json"])
except Exception as e:
    print("warn: base repo prefetch:", e)
print("Human-1 weights ready in", c["weights_dir"])
PY
fi

if [ "$VEENA_ON" = "1" ]; then
    echo "Downloading Veena + SNAC weights..."
    python3 - <<'PY'
import json
from huggingface_hub import snapshot_download
c = json.load(open("config.json"))["veena"]
snapshot_download(repo_id=c["hf_repo"], local_dir=c["weights_dir"])
try:
    snapshot_download(repo_id=c["snac_repo"])  # cached for from_pretrained
except Exception as e:
    print("warn: snac prefetch:", e)
print("Veena weights ready in", c["weights_dir"])
PY
fi

if [ "$SVARA_ON" = "1" ]; then
    echo "Downloading Svara + SNAC weights..."
    python3 - <<'PY'
import json
from huggingface_hub import snapshot_download
c = json.load(open("config.json"))["svara"]
snapshot_download(repo_id=c["hf_repo"], local_dir=c["weights_dir"])
try:
    snapshot_download(repo_id=c["snac_repo"])  # cached for from_pretrained
except Exception as e:
    print("warn: snac prefetch:", e)
print("Svara weights ready in", c["weights_dir"])
PY
fi

echo "=================================================================="
echo " Setup complete. Start the server with:  python3 server.py"
echo "=================================================================="
