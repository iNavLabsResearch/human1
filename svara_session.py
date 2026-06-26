"""
svara_session.py
================
Streaming Text-to-Speech over a WebSocket for Svara-TTS v1 (Orpheus-style).

Pipeline:
    "{voice}: {text}" --> [SOH] ids [EOT][EOH] --> Svara LLM (streamed audio
    tokens) --> de-interleave 7 tokens/frame --> SNAC decode (sliding 4-frame
    window) --> Int16 PCM @ 24 kHz streamed to the browser.

Reported metrics (per request, not per chunk):
    FCL  : first-chunk latency — model time to the first audio chunk
    RTF  : real-time factor = total generation time / produced audio duration

Wire protocol
-------------
Client -> server:
    {"type":"tts","req_id":str,"text":str,"voice":str,"temperature":float,"top_p":float}

Server -> client:
    text {"type":"begin","req_id":str,"server_ts":ms}
    text {"type":"chunk","req_id":str,"index":int,"gen_ms":float,"samples":int}  <- header
    bin  Int16-LE mono PCM @ 24 kHz                                              <- follows header
    text {"type":"end","req_id":str,"chunks":int,"total_ms":float,"fcl_ms":float,
          "audio_sec":float,"rtf":float,"total_samples":int}
    text {"type":"error","req_id":str,"message":str}
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import Any, Iterator, Optional

import numpy as np


def _make_token_streamer():
    from transformers.generation.streamers import BaseStreamer

    class TokenStreamer(BaseStreamer):
        def __init__(self) -> None:
            self.q: "queue.Queue[Optional[int]]" = queue.Queue()
            self._skip_prompt = True

        def put(self, value) -> None:
            if self._skip_prompt:
                self._skip_prompt = False
                return
            if hasattr(value, "shape") and len(value.shape) > 1:
                value = value[0]
            for t in value.tolist():
                self.q.put(int(t))

        def end(self) -> None:
            self.q.put(None)

    return TokenStreamer()


def _build_input_ids(cache, text: str, voice: str):
    """Orpheus prompt: [SOH] + tok('{voice}: {text}') + [EOT, EOH]."""
    import torch

    T = cache.tokens
    prompt = f"{voice}: {text}"
    ids = cache.tokenizer.encode(prompt)  # Llama3 tokenizer adds BOS by default
    input_tokens = [T["start_of_human"], *ids, T["end_of_text"], T["end_of_human"]]
    return torch.tensor([input_tokens], device=cache.model.device)


def _decode_window(codes28: list[int], snac, snac_device):
    import torch

    l0, l1, l2 = [], [], []
    for j in range(4):
        i = 7 * j
        l0.append(codes28[i])
        l1.append(codes28[i + 1]); l1.append(codes28[i + 4])
        l2.append(codes28[i + 2]); l2.append(codes28[i + 3])
        l2.append(codes28[i + 5]); l2.append(codes28[i + 6])
    codes = [
        torch.tensor(l0, dtype=torch.int32, device=snac_device).unsqueeze(0),
        torch.tensor(l1, dtype=torch.int32, device=snac_device).unsqueeze(0),
        torch.tensor(l2, dtype=torch.int32, device=snac_device).unsqueeze(0),
    ]
    for t in codes:
        if torch.any((t < 0) | (t > 4095)):
            return None
    with torch.no_grad():
        audio = snac.decode(codes)
    audio = audio[:, :, 2048:4096].squeeze().clamp(-1, 1).cpu().numpy()
    return (audio * 32767.0).astype("<i2").tobytes()


def synth_blocking(cache, text: str, voice: str,
                   temperature: float = 0.6, top_p: float = 0.95) -> Iterator[bytes]:
    """Blocking generator yielding Int16 PCM chunks as they are produced."""
    import torch

    gcfg = cache.config["svara"]["generation"]
    T = cache.tokens
    base = T["audio_code_base_offset"]

    input_ids = _build_input_ids(cache, text, voice)

    streamer = _make_token_streamer()
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=int(gcfg.get("max_new_tokens", 1200)),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(gcfg.get("repetition_penalty", 1.1)),
        eos_token_id=T["end_of_speech"],
        pad_token_id=T.get("pad", cache.tokenizer.pad_token_id),
        streamer=streamer,
    )

    thread = threading.Thread(target=_safe_generate, args=(cache.model, gen_kwargs), daemon=True)
    thread.start()

    buffer: list[int] = []
    audio_count = 0
    while True:
        tok = streamer.q.get()
        if tok is None:
            break
        if tok in (T["end_of_speech"],):
            break
        slot = audio_count % 7
        code = tok - base - slot * 4096
        if 0 <= code < 4096:
            buffer.append(code)
            audio_count += 1
            if audio_count % 7 == 0 and audio_count >= 28:
                chunk = _decode_window(buffer[-28:], cache.snac, cache.snac_device)
                if chunk:
                    yield chunk
    thread.join(timeout=1.0)


def _safe_generate(model, kwargs) -> None:
    import torch

    try:
        with torch.no_grad():
            model.generate(**kwargs)
    except Exception as exc:
        print(f"[svara] generate error: {exc}")
        try:
            kwargs["streamer"].end()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
class SvaraSession:
    def __init__(self, websocket, cache) -> None:
        self.ws = websocket
        self.cache = cache
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._out_q: "asyncio.Queue[Any]" = asyncio.Queue()

    async def run(self) -> None:
        await self.ws.accept()
        self._loop = asyncio.get_running_loop()
        send_task = asyncio.create_task(self._send_loop())
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is None:
                    continue
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue
                if data.get("type") == "tts":
                    await self._handle_tts(data)
                elif data.get("type") == "ping":
                    await self._out_q.put({"type": "pong", "t": data.get("t"),
                                           "server_t": time.time() * 1000.0})
        finally:
            await self._out_q.put(None)
            await send_task
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _handle_tts(self, data: dict[str, Any]) -> None:
        req_id = data.get("req_id", "req")
        text = (data.get("text") or "").strip()
        voice = data.get("voice") or self.cache.config["svara"].get("default_voice")
        temperature = float(data.get("temperature", 0.6))
        top_p = float(data.get("top_p", 0.95))

        if not text:
            await self._out_q.put({"type": "error", "req_id": req_id, "message": "empty text"})
            return
        if voice not in self.cache.voices:
            voice = self.cache.config["svara"].get("default_voice") or (self.cache.voices[0] if self.cache.voices else "Hindi (Female)")

        await self._out_q.put({"type": "begin", "req_id": req_id, "voice": voice,
                               "server_ts": time.time() * 1000.0})
        await asyncio.to_thread(self._run_synth, req_id, text, voice, temperature, top_p)

    def _run_synth(self, req_id, text, voice, temperature, top_p) -> None:
        t_start = time.perf_counter()
        index = 0
        fcl_ms = None
        total_samples = 0
        try:
            for chunk in synth_blocking(self.cache, text, voice, temperature, top_p):
                now = time.perf_counter()
                gen_ms = (now - t_start) * 1000.0
                if fcl_ms is None:
                    fcl_ms = gen_ms
                n_samples = len(chunk) // 2
                total_samples += n_samples
                self._emit({"type": "chunk", "req_id": req_id, "index": index,
                            "gen_ms": round(gen_ms, 2), "samples": n_samples})
                self._emit(chunk)
                index += 1

            total_ms = (time.perf_counter() - t_start) * 1000.0
            audio_sec = total_samples / float(self.cache.sample_rate)
            rtf = round((total_ms / 1000.0) / audio_sec, 3) if audio_sec > 0 else None
            self._emit({"type": "end", "req_id": req_id, "chunks": index,
                        "total_ms": round(total_ms, 2),
                        "fcl_ms": round(fcl_ms, 2) if fcl_ms is not None else None,
                        "audio_sec": round(audio_sec, 3),
                        "rtf": rtf, "total_samples": total_samples})
        except Exception as exc:
            self._emit({"type": "error", "req_id": req_id,
                        "message": f"{type(exc).__name__}: {exc}"})

    def _emit(self, item: Any) -> None:
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._out_q.put_nowait, item)
        except RuntimeError:
            pass

    async def _send_loop(self) -> None:
        while True:
            item = await self._out_q.get()
            if item is None:
                break
            try:
                if isinstance(item, (bytes, bytearray)):
                    await self.ws.send_bytes(item)
                else:
                    await self.ws.send_text(json.dumps(item))
            except Exception:
                break
