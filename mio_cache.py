"""
mio_cache.py
============
Load-once singleton for the Indic-Mio TTS stack:

    * Indic-Mio LLM (SPRINGLab/Indic-Mio — Qwen3-0.6B, base Aratako/MioTTS-0.6B)
    * MioCodec     (Aratako/MioCodec-25Hz-24kHz) to turn flat audio codes -> PCM

Unlike Svara/Veena this is NOT Orpheus/SNAC: it uses a Qwen3 chat-template
prompt and a single flat codebook (audio token = id - speech_offset, valid for
``audio_vocab_size`` ids). Voice is zero-shot (speaker embeddings); the model
exposes no named voice IDs, so the UI selects language + emotion instead.

Heavy imports are lazy so the FastAPI app imports fine on a GPU-less box.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional


class MioCache:
    def __init__(self) -> None:
        self.loaded = False
        self.loading = False
        self.load_error: Optional[str] = None
        self.load_seconds = 0.0

        self.config: dict[str, Any] = {}
        self.model = None
        self.tokenizer = None
        self.codec = None
        self.device = None
        self.codec_device = None

        self.sample_rate = 24000
        self.languages: list[str] = []

        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def load(self, config: dict[str, Any]) -> None:
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
                print(f"[mio] ready in {self.load_seconds:.1f}s on {self.device} "
                      f"(sr={self.sample_rate}, quant={config['mio']['quantization']})")
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                print(f"[mio] LOAD FAILED: {self.load_error}")
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
            "languages": self.languages,
        }

    # ------------------------------------------------------------------ #
    def _load_impl(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        mcfg = config["mio"]
        self.languages = mcfg.get("languages", [])

        want_device = mcfg.get("device", "cuda")
        if want_device == "cuda" and not torch.cuda.is_available():
            print("[mio] CUDA unavailable -> CPU")
            want_device = "cpu"
        self.device = torch.device(want_device)

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(mcfg.get("dtype", "bfloat16"), torch.bfloat16)
        if self.device.type == "cpu":
            dtype = torch.float32

        src = mcfg.get("weights_dir") or mcfg["hf_repo"]
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}

        quant = mcfg.get("quantization", "none").lower()
        if self.device.type == "cuda" and quant in ("4bit", "int8"):
            from transformers import BitsAndBytesConfig

            if quant == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = mcfg.get("device_map", "auto")
        else:
            model_kwargs["torch_dtype"] = dtype
            if self.device.type == "cuda":
                model_kwargs["device_map"] = mcfg.get("device_map", "auto")

        print(f"[mio] loading LLM from {src} ...")
        self.model = AutoModelForCausalLM.from_pretrained(src, **model_kwargs)
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

        print("[mio] loading tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

        print(f"[mio] loading MioCodec ({mcfg['codec_repo']}) ...")
        from miocodec import MioCodec

        self.codec = MioCodec.from_pretrained(mcfg["codec_repo"])
        try:
            self.codec = self.codec.eval()
        except Exception:
            pass
        if self.device.type == "cuda":
            try:
                self.codec = self.codec.to(self.device)
            except Exception as exc:
                print(f"[mio] codec.to(cuda) failed ({exc}); leaving on default device")

        # resolve true codec output sample rate (codec name says 24kHz)
        sr = (getattr(self.codec, "sample_rate", None)
              or getattr(self.codec, "sampling_rate", None)
              or getattr(self.codec, "sr", None))
        self.sample_rate = int(sr) if sr else int(mcfg.get("sample_rate", 24000))

        # device the codec actually sits on (for building code tensors)
        try:
            self.codec_device = next(self.codec.parameters()).device
        except Exception:
            self.codec_device = self.device

        if mcfg.get("warmup", True):
            self._warmup()

    def _warmup(self) -> None:
        try:
            from mio_session import synth_blocking

            print("[mio] warming up ...")
            list(synth_blocking(self, "नमस्ते।"))
        except Exception as exc:
            print(f"[mio] warmup skipped ({exc})")


CACHE = MioCache()
