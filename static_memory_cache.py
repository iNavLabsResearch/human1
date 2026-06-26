"""
static_memory_cache.py
======================
Process-wide singleton that loads the Human-1 (Moshi) model **once** on server
startup and keeps it resident in (GPU) memory for the lifetime of the process.

Everything is driven by ``config.json`` — nothing is hard-coded here.

Exposed objects after ``CACHE.load(config)``:
    CACHE.mimi            -> Mimi audio codec (frozen, 24kHz / 12.5Hz)
    CACHE.lm              -> Moshi language model (quantized per config)
    CACHE.text_tokenizer  -> Hindi SentencePiece tokenizer
    CACHE.device          -> torch.device the model lives on
    CACHE.frame_size      -> samples per Mimi frame (sample_rate / frame_rate)
    CACHE.config          -> the parsed config dict

This module imports torch/moshi lazily inside ``load()`` so the FastAPI app can
be imported on a machine without a GPU (for linting / inspection).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional


class StaticMemoryCache:
    """Lazy, thread-safe, load-once cache for the heavy model objects."""

    def __init__(self) -> None:
        self.loaded: bool = False
        self.loading: bool = False
        self.load_error: Optional[str] = None
        self.load_seconds: float = 0.0

        self.config: dict[str, Any] = {}
        self.device = None
        self.dtype = None

        self.mimi = None
        self.lm = None
        self.text_tokenizer = None

        self.sample_rate: int = 24000
        self.frame_rate: float = 12.5
        self.frame_size: int = 1920

        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self, config: dict[str, Any]) -> None:
        """Load the model into memory. Safe to call once at startup."""
        with self._lock:
            if self.loaded:
                return
            self.loading = True
            self.load_error = None
            t0 = time.time()
            try:
                self._load_impl(config)
                self.loaded = True
                self.load_seconds = time.time() - t0
                print(f"[cache] model ready in {self.load_seconds:.1f}s "
                      f"on {self.device} (quant={config['model']['quantization']})")
            except Exception as exc:  # surface a readable error to /health
                self.load_error = f"{type(exc).__name__}: {exc}"
                print(f"[cache] LOAD FAILED: {self.load_error}")
                raise
            finally:
                self.loading = False

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "error": self.load_error,
            "load_seconds": round(self.load_seconds, 2),
            "device": str(self.device) if self.device else None,
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "frame_size": self.frame_size,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _load_impl(self, config: dict[str, Any]) -> None:
        import torch  # noqa: WPS433  (lazy, heavy)
        from moshi.models import loaders

        self.config = config
        mcfg = config["human1"]

        # --- device + dtype -------------------------------------------------
        want_device = mcfg.get("device", "cuda")
        if want_device == "cuda" and not torch.cuda.is_available():
            print("[cache] CUDA not available -> falling back to CPU")
            want_device = "cpu"
        self.device = torch.device(want_device)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(mcfg.get("dtype", "bfloat16"), torch.bfloat16)
        if self.device.type == "cpu":
            self.dtype = torch.float32  # bf16 matmul is slow/unsupported on CPU

        # --- resolve local weight paths ------------------------------------
        wdir = Path(mcfg["weights_dir"])
        moshi_w = str(wdir / mcfg["moshi_weight"])
        mimi_w = str(wdir / mcfg["mimi_weight"])
        tok = str(wdir / mcfg["tokenizer"])

        for p in (moshi_w, mimi_w, tok):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"missing weight file: {p} — run ./setup.sh first")

        # --- build CheckpointInfo ------------------------------------------
        # Human-1 ships no config.json, so we borrow the architecture/config
        # defaults from the base Moshi repo and point the weights at Human-1.
        print("[cache] building CheckpointInfo ...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            mcfg["base_repo"],
            moshi_weights=moshi_w,
            mimi_weights=mimi_w,
            tokenizer=tok,
            config_path=mcfg.get("config_path"),
        )

        # --- Mimi codec (always full precision, it is tiny + frozen) --------
        print("[cache] loading Mimi codec ...")
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.sample_rate = int(self.mimi.sample_rate)
        self.frame_rate = float(self.mimi.frame_rate)
        self.frame_size = int(self.sample_rate / self.frame_rate)

        # --- Moshi LM -------------------------------------------------------
        print("[cache] loading Moshi LM ...")
        self.lm = checkpoint_info.get_moshi(device=self.device, dtype=self.dtype)

        # --- quantization ---------------------------------------------------
        quant = mcfg.get("quantization", "none").lower()
        if quant == "int8":
            self._quantize_int8(self.lm)

        # --- multi-GPU dispatch (optional) ----------------------------------
        self._maybe_dispatch_multi_gpu(mcfg)

        self.lm.eval()

        # --- text tokenizer -------------------------------------------------
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()

        # --- warmup (build CUDA graphs / autotune) --------------------------
        self._warmup(int(mcfg.get("warmup_steps", 4)))

    def _quantize_int8(self, lm) -> None:
        """INT8 weight-only quantization of the LM's Linear layers.

        Tries torchao first (best perf, keeps activations in bf16), then
        bitsandbytes, then a torch dynamic-quant fallback (CPU). All are
        best-effort: on failure we keep the model at its loaded dtype and warn.
        """
        import torch

        # 1) torchao int8 weight-only -- the preferred path on GPU.
        try:
            from torchao.quantization import quantize_, int8_weight_only

            quantize_(lm, int8_weight_only())
            print("[cache] INT8 quantization applied via torchao "
                  "(int8_weight_only)")
            return
        except Exception as exc:
            print(f"[cache] torchao int8 unavailable ({exc}); trying fallback")

        # 2) torch dynamic quantization -- CPU only, last resort.
        try:
            if self.device.type == "cpu":
                torch.quantization.quantize_dynamic(
                    lm, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
                print("[cache] INT8 dynamic quantization applied (torch, CPU)")
                return
        except Exception as exc:
            print(f"[cache] torch dynamic quant failed ({exc})")

        print("[cache] WARNING: INT8 requested but no backend succeeded; "
              f"running at {self.dtype}.")

    def _maybe_dispatch_multi_gpu(self, mcfg: dict[str, Any]) -> None:
        """Spread the LM across multiple GPUs when device_map != 'single'.

        On a multi-GPU box (e.g. Kaggle 2x T4) accelerate balances the LM's
        layers across devices. Mimi stays on the primary device. This keeps
        the streaming inference path working while using all available VRAM.
        """
        import torch

        device_map = mcfg.get("device_map", "single")
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if device_map == "single" or n <= 1:
            return
        try:
            from accelerate import dispatch_model, infer_auto_device_map

            amap = infer_auto_device_map(self.lm)
            self.lm = dispatch_model(self.lm, device_map=amap)
            print(f"[cache] LM dispatched across {n} GPUs (device_map={device_map})")
        except Exception as exc:
            print(f"[cache] multi-GPU dispatch failed ({exc}); "
                  "keeping LM on single device")

    def _warmup(self, steps: int) -> None:
        import torch

        if steps <= 0:
            return
        print(f"[cache] warming up ({steps} steps) ...")
        gen = config_lmgen(self.lm, self.config)
        try:
            with torch.no_grad(), self.mimi.streaming(1), gen.streaming(1):
                zeros = torch.zeros(
                    1, 1, self.frame_size, device=self.device, dtype=torch.float32)
                for _ in range(steps):
                    codes = self.mimi.encode(zeros)
                    for t in range(codes.shape[-1]):
                        gen.step(codes[:, :, t:t + 1])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as exc:
            print(f"[cache] warmup skipped ({exc})")


def config_lmgen(lm, config: dict[str, Any]):
    """Build a fresh LMGen (per-connection generation state) from config."""
    from moshi.models import LMGen

    g = config["human1"]["generation"]
    return LMGen(
        lm,
        use_sampling=g.get("use_sampling", True),
        temp=g.get("temp", 0.8),
        temp_text=g.get("temp_text", 0.7),
        top_k=g.get("top_k", 250),
        top_k_text=g.get("top_k_text", 25),
    )


# Process-wide singleton imported by the FastAPI app.
CACHE = StaticMemoryCache()
