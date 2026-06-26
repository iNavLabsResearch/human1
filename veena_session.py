"""
veena_session.py
================
Streaming Text-to-Speech over a WebSocket for the Veena model.

Pipeline (Orpheus/SNAC streaming scheme):
    text --> Veena LLM (autoregressive audio tokens, streamed) -->
    de-interleave 7 tokens/frame --> SNAC decode (sliding 4-frame window) -->
    Int16 PCM @ 24 kHz pushed to the browser as it is produced.

Per audio chunk the server reports a latency breakdown:
    gen_ms     : model time since the request started (cumulative)
    gen_delta  : model time for *this* chunk
    server_ms  : total server handling time since request received (~ gen_ms)

The client adds:
    client_ms  : wall time from "text submitted" to "chunk received"

Wire protocol
-------------
Client -> server (text JSON):
    {"type":"tts","req_id":str,"text":str,"speaker":str,
     "temperature":float,"top_p":float}

Server -> client:
    text {"type":"begin","req_id":str,"server_ts":ms}
    text {"type":"chunk","req_id":str,"index":int,"gen_ms":..,"gen_delta":..,
          "server_ms":..,"server_ts":ms,"samples":int}   <-- header
    bin  Int16-LE mono PCM @ 24 kHz                        <-- the chunk (follows header)
    text {"type":"end","req_id":str,"total_ms":..,"chunks":int,"gen_ms":..}
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


# --------------------------------------------------------------------------- #
# Token streaming from HF generate
# --------------------------------------------------------------------------- #
def _make_token_streamer():
    from transformers.generation.streamers import BaseStreamer

    class TokenStreamer(BaseStreamer):
        """Pushes generated token ids onto a queue, skipping the prompt."""

        def __init__(self) -> None:
            self.q: "queue.Queue[Optional[int]]" = queue.Queue()
            self._skip_prompt = True

        def put(self, value) -> None:
            if self._skip_prompt:           # first call carries the prompt
                self._skip_prompt = False
                return
            if hasattr(value, "shape") and len(value.shape) > 1:
                value = value[0]
            for t in value.tolist():
                self.q.put(int(t))

        def end(self) -> None:
            self.q.put(None)

    return TokenStreamer()


def _build_input_ids(cache, text: str, speaker: str):
    import torch

    tok = cache.tokenizer
    T = cache.tokens
    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = tok.encode(prompt, add_special_tokens=False)
    input_tokens = [
        T["start_of_human"],
        *prompt_tokens,
        T["end_of_human"],
        T["start_of_ai"],
        T["start_of_speech"],
    ]
    return torch.tensor([input_tokens], device=cache.model.device), len(text)


def _decode_window(codes28: list[int], snac, snac_device):
    """Decode a 4-frame (28-token) window and return the fresh middle slice."""
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
    # middle slice keeps the streamed output gapless (Orpheus scheme)
    audio = audio[:, :, 2048:4096].squeeze().clamp(-1, 1).cpu().numpy()
    return (audio * 32767.0).astype("<i2").tobytes()


def synth_blocking(cache, text: str, speaker: str,
                   temperature: float = 0.4, top_p: float = 0.9) -> Iterator[bytes]:
    """Generator yielding Int16 PCM chunks as they are produced (blocking)."""
    import torch

    vcfg = cache.config["veena"]
    gcfg = vcfg["generation"]
    T = cache.tokens
    base = T["audio_code_base_offset"]

    input_ids, text_len = _build_input_ids(cache, text, speaker)
    max_new = min(int(text_len * 1.3) * 7 + 21, int(gcfg.get("max_new_tokens", 700)))

    streamer = _make_token_streamer()
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(gcfg.get("repetition_penalty", 1.05)),
        pad_token_id=cache.tokenizer.pad_token_id,
        eos_token_id=[T["end_of_speech"], T["end_of_ai"]],
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
        if tok in (T["end_of_speech"], T["end_of_ai"]):
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
    except Exception as exc:  # ensure the streamer is closed so the loop ends
        print(f"[veena] generate error: {exc}")
        try:
            kwargs["streamer"].end()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# WebSocket session
# --------------------------------------------------------------------------- #
class VeenaSession:
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
        speaker = data.get("speaker") or self.cache.config["veena"]["default_speaker"]
        temperature = float(data.get("temperature", 0.4))
        top_p = float(data.get("top_p", 0.9))

        if not text:
            await self._out_q.put({"type": "error", "req_id": req_id, "message": "empty text"})
            return
        if speaker not in self.cache.speakers:
            speaker = self.cache.config["veena"]["default_speaker"]

        await self._out_q.put({"type": "begin", "req_id": req_id,
                               "server_ts": time.time() * 1000.0})
        # run blocking generation in a worker thread, stream chunks out
        await asyncio.to_thread(self._run_synth, req_id, text, speaker, temperature, top_p)

    def _run_synth(self, req_id, text, speaker, temperature, top_p) -> None:
        t_start = time.perf_counter()
        last = t_start
        index = 0
        try:
            for chunk in synth_blocking(self.cache, text, speaker, temperature, top_p):
                now = time.perf_counter()
                gen_ms = (now - t_start) * 1000.0
                gen_delta = (now - last) * 1000.0
                last = now
                header = {
                    "type": "chunk", "req_id": req_id, "index": index,
                    "gen_ms": round(gen_ms, 2),
                    "gen_delta": round(gen_delta, 2),
                    "server_ms": round(gen_ms, 2),  # server overhead ~= gen time
                    "server_ts": time.time() * 1000.0,
                    "samples": len(chunk) // 2,
                }
                self._emit(header)
                self._emit(chunk)
                index += 1
            total_ms = (time.perf_counter() - t_start) * 1000.0
            self._emit({"type": "end", "req_id": req_id, "chunks": index,
                        "total_ms": round(total_ms, 2), "gen_ms": round(total_ms, 2)})
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
