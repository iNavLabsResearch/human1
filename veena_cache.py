"""
veena_cache.py
==============
Process-wide singleton that loads the Veena TTS stack **once** on startup:

    * Veena LLM  (maya-research/Veena — Llama-3B, autoregressive audio-token LM)
    * SNAC 24kHz neural codec (hubertsiuzdak/snac_24khz) to turn codes -> PCM

Quantization is config-driven: ``4bit`` (nf4, default, per model card), ``int8``
(bitsandbytes 8-bit) or ``none``. Everything else lives in ``config["veena"]``.

torch / transformers / snac are imported lazily inside ``load()`` so the FastAPI
app can be imported on a machine without a GPU.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional


class VeenaCache:
    def __init__(self) -> None:
        self.loaded = False
        self.loading = False
        self.load_error: Optional[str] = None
        self.load_seconds = 0.0

        self.config: dict[str, Any] = {}
        self.model = None
        self.tokenizer = None
        self.snac = None
        self.device = None
        self.snac_device = None

        self.sample_rate = 24000
        self.speakers: list[str] = []
        self.tokens: dict[str, int] = {}

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
                print(f"[veena] ready in {self.load_seconds:.1f}s on {self.device} "
                      f"(quant={config['veena']['quantization']})")
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                print(f"[veena] LOAD FAILED: {self.load_error}")
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
            "speakers": self.speakers,
        }

    # ------------------------------------------------------------------ #
    def _load_impl(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        vcfg = config["veena"]
        self.speakers = vcfg.get("speakers", [])
        self.tokens = vcfg["tokens"]
        self.sample_rate = int(vcfg.get("sample_rate", 24000))

        want_device = vcfg.get("device", "cuda")
        if want_device == "cuda" and not torch.cuda.is_available():
            print("[veena] CUDA unavailable -> CPU (quantization disabled)")
            want_device = "cpu"
        self.device = torch.device(want_device)

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(vcfg.get("dtype", "bfloat16"), torch.bfloat16)

        src = vcfg.get("weights_dir") or vcfg["hf_repo"]
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}

        quant = vcfg.get("quantization", "none").lower()
        if self.device.type == "cuda" and quant in ("4bit", "int8"):
            from transformers import BitsAndBytesConfig

            if quant == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = vcfg.get("device_map", "auto")
        else:
            model_kwargs["torch_dtype"] = dtype
            if self.device.type == "cuda":
                model_kwargs["device_map"] = vcfg.get("device_map", "auto")

        print(f"[veena] loading LLM from {src} ...")
        self.model = AutoModelForCausalLM.from_pretrained(src, **model_kwargs)
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

        print("[veena] loading tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

        print(f"[veena] loading SNAC codec ({vcfg['snac_repo']}) ...")
        from snac import SNAC

        self.snac = SNAC.from_pretrained(vcfg["snac_repo"]).eval()
        if self.device.type == "cuda":
            self.snac = self.snac.to(self.device)
        self.snac_device = next(self.snac.parameters()).device

        if vcfg.get("warmup", True):
            self._warmup()

    def _warmup(self) -> None:
        try:
            from veena_session import synth_blocking

            print("[veena] warming up ...")
            list(synth_blocking(self, "नमस्ते", self.config["veena"]["default_speaker"]))
        except Exception as exc:
            print(f"[veena] warmup skipped ({exc})")


CACHE = VeenaCache()
