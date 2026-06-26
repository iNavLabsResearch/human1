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
read -r HUMAN1_ON VEENA_ON SVARA_ON MIO_ON <<<"$(python3 - <<'PY'
import json
c = json.load(open("config.json"))
print(int(c.get("human1", {}).get("enabled", False)),
      int(c.get("veena", {}).get("enabled", False)),
      int(c.get("svara", {}).get("enabled", False)),
      int(c.get("mio", {}).get("enabled", False)))
PY
)"
echo "human1 enabled: $HUMAN1_ON"
echo "veena  enabled: $VEENA_ON"
echo "svara  enabled: $SVARA_ON"
echo "mio    enabled: $MIO_ON"

# ----------------------------------------------------------------------------
# PyTorch (CUDA build). Skip if torch already present (Kaggle/Colab images).
# ----------------------------------------------------------------------------
python3 -m pip install --upgrade pip wheel

# ----------------------------------------------------------------------------
# PyTorch — must match the GPU arch. Blackwell (RTX 6000 Blackwell / B200 /
# 50-series = sm_120) needs a CUDA 12.8 build; older cu121 wheels have no
# sm_120 kernels and crash at runtime. We (re)install from cu128 unless the
# torch already present actually lists this GPU's arch.
# ----------------------------------------------------------------------------
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
NEED_TORCH=1
if python3 -c "import torch" 2>/dev/null; then
    if python3 - <<'PY'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        sys.exit(1)
    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    print(f"[torch] {torch.__version__} gpu={torch.cuda.get_device_name(0)} "
          f"cap={sm} arch_list={archs}")
    sys.exit(0 if sm in archs else 2)
except Exception as e:
    print("[torch] check failed:", e); sys.exit(1)
PY
    then
        NEED_TORCH=0
        echo "torch already supports this GPU."
    else
        echo "Installed torch does NOT support this GPU arch -> reinstalling from cu128."
    fi
fi
if [ "$NEED_TORCH" = "1" ]; then
    echo "Installing torch + torchaudio from $TORCH_INDEX ..."
    python3 -m pip install --upgrade --force-reinstall torch torchaudio --index-url "$TORCH_INDEX"
fi

# ----------------------------------------------------------------------------
# Python deps (fastapi, pyngrok always; model deps as needed)
# ----------------------------------------------------------------------------
echo "Installing Python requirements..."
python3 -m pip install -r requirements.txt

# miocodec is git-only (not on PyPI); install it when Indic-Mio is enabled.
if [ "$MIO_ON" = "1" ]; then
    echo "Installing miocodec from git ..."
    python3 -m pip install "git+https://github.com/Aratako/MioCodec"
fi

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

if [ "$MIO_ON" = "1" ]; then
    echo "Downloading Indic-Mio (incl. Indian sample voices) + MioCodec weights..."
    python3 - <<'PY'
import json, os
from huggingface_hub import snapshot_download
c = json.load(open("config.json"))["mio"]
# whole repo -> includes samples/*.wav (the Indian reference voices) + tokenizer
snapshot_download(repo_id=c["hf_repo"], local_dir=c["weights_dir"])
try:
    snapshot_download(repo_id=c["codec_repo"])  # cached for MioCodecModel.from_pretrained
except Exception as e:
    print("warn: codec prefetch:", e)
os.makedirs(c.get("presets_dir", "./mio_presets"), exist_ok=True)  # for user .pt voices
ref_dir = c.get("reference_dir", "")
print("Indic-Mio ready in", c["weights_dir"], "; voice refs in", ref_dir)
print("voice refs present:", [f for f in (c.get("voices") or {}).values()
                              if os.path.exists(os.path.join(ref_dir, f))])
PY
fi

echo "=================================================================="
echo " Setup complete. Start the server with:  python3 server.py"
echo "=================================================================="
