"""
svara_cache.py
==============
Load-once singleton for the Svara-TTS v1 stack:

    * Svara LLM   (kenpath/svara-tts-v1 — Orpheus-style Llama-3B audio-token LM,
                   base canopylabs/3b-hi-ft-research_release)
    * SNAC 24kHz codec (hubertsiuzdak/snac_24khz) to turn codes -> PCM

Voices follow the convention "Language (Gender)" (e.g. "Hindi (Female)"); the
full voice list is built from ``config["svara"]["languages"] x genders``.

Quantization is config-driven (``4bit`` default / ``int8`` / ``none``). Heavy
imports are lazy so the FastAPI app imports fine on a GPU-less box.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional


class SvaraCache:
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
        self.voices: list[str] = []
        self.languages: list[str] = []
        self.genders: list[str] = []
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
                print(f"[svara] ready in {self.load_seconds:.1f}s on {self.device} "
                      f"(quant={config['svara']['quantization']})")
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                print(f"[svara] LOAD FAILED: {self.load_error}")
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
            "genders": self.genders,
            "voices": self.voices,
        }

    # ------------------------------------------------------------------ #
    def _load_impl(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        scfg = config["svara"]
        self.languages = scfg.get("languages", [])
        self.genders = scfg.get("genders", ["Female", "Male"])
        self.voices = [f"{lang} ({g})" for lang in self.languages for g in self.genders]
        self.tokens = scfg["tokens"]
        self.sample_rate = int(scfg.get("sample_rate", 24000))

        want_device = scfg.get("device", "cuda")
        if want_device == "cuda" and not torch.cuda.is_available():
            print("[svara] CUDA unavailable -> CPU (quantization disabled)")
            want_device = "cpu"
        self.device = torch.device(want_device)

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(scfg.get("dtype", "bfloat16"), torch.bfloat16)

        src = scfg.get("weights_dir") or scfg["hf_repo"]
        model_kwargs: dict[str, Any] = {}

        quant = scfg.get("quantization", "none").lower()
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
            model_kwargs["device_map"] = scfg.get("device_map", "auto")
        else:
            model_kwargs["torch_dtype"] = dtype
            if self.device.type == "cuda":
                model_kwargs["device_map"] = scfg.get("device_map", "auto")

        print(f"[svara] loading LLM from {src} ...")
        self.model = AutoModelForCausalLM.from_pretrained(src, **model_kwargs)
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

        print("[svara] loading tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(src)

        print(f"[svara] loading SNAC codec ({scfg['snac_repo']}) ...")
        from snac import SNAC

        self.snac = SNAC.from_pretrained(scfg["snac_repo"]).eval()
        if self.device.type == "cuda":
            self.snac = self.snac.to(self.device)
        self.snac_device = next(self.snac.parameters()).device

        if scfg.get("warmup", True):
            self._warmup()

    def _warmup(self) -> None:
        try:
            from svara_session import synth_blocking

            print("[svara] warming up ...")
            voice = self.config["svara"].get("default_voice") or (self.voices[0] if self.voices else "Hindi (Female)")
            list(synth_blocking(self, "नमस्ते।", voice))
        except Exception as exc:
            print(f"[svara] warmup skipped ({exc})")


CACHE = SvaraCache()
