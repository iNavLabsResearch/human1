"""
mio_session.py
==============
Streaming Text-to-Speech over a WebSocket for Indic-Mio (Qwen3 + MioCodec).

Pipeline:
    chat_template(user=text [+ <emotion>]) --> Indic-Mio LLM (streamed tokens)
    --> keep audio tokens (id - speech_offset in [0, audio_vocab_size)) -->
    MioCodec.decode (flat single codebook) --> Int16 PCM streamed to the browser.

MioCodec is a flat 25 Hz codec (no 7-token framing). We stream by decoding the
accumulated codes with full context every ``decode_every`` new tokens and
emitting only the freshly-produced tail samples.

Reported per request: FCL (first-chunk latency) and RTF.

Wire protocol
-------------
Client -> server:
    {"type":"tts","req_id":str,"text":str,"voice":str,"language":str,
     "emotion":str,"temperature":float,"top_p":float}
    # voice = a MioTTS speaker preset name (en_female / en_male / jp_female / jp_male)

Server -> client:
    text {"type":"begin","req_id":str,"sample_rate":int,"server_ts":ms}
    text {"type":"chunk","req_id":str,"index":int,"gen_ms":float,"samples":int}  <- header
    bin  Int16-LE mono PCM @ sample_rate                                         <- follows header
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


def _build_inputs(cache, text: str, emotion: Optional[str]):
    """Qwen3 chat-template prompt; emotion tag appended at the end of the text."""
    content = text.strip()
    if emotion:
        content = f"{content} <{emotion}>"
    messages = [{"role": "user", "content": content}]
    prompt = cache.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return cache.tokenizer(prompt, return_tensors="pt").to(cache.model.device)


def _decode(cache, codes: list[int], global_embedding):
    """Decode flat MioCodec content tokens -> float32 PCM (full context).

    MioCodecModel.decode(global_embedding=<1D speaker emb>,
                         content_token_indices=<1D long>, target_audio_length=None)
    """
    import torch

    if not codes or global_embedding is None:
        return None
    idx = torch.tensor(codes, dtype=torch.long, device=cache.codec_device)  # (seq_len,)
    with torch.no_grad():
        wav = cache.codec.decode(global_embedding=global_embedding,
                                 content_token_indices=idx)
    return wav.squeeze().clamp(-1, 1).float().cpu().numpy()


def _to_i16(pcm: np.ndarray) -> bytes:
    return (np.asarray(pcm) * 32767.0).astype("<i2").tobytes()


def synth_blocking(cache, text: str, emotion: Optional[str] = None,
                   voice: Optional[str] = None, global_embedding=None,
                   temperature: float = 0.9, top_p: float = 0.9) -> Iterator[bytes]:
    """Blocking generator yielding Int16 PCM chunks as they are produced."""
    import torch

    mcfg = cache.config["mio"]
    gcfg = mcfg["generation"]
    offset = int(mcfg["speech_offset"])
    vsize = int(mcfg["audio_vocab_size"])
    eos_ids = set(mcfg.get("eos_token_ids", [151645, 151643]))
    decode_every = int(mcfg.get("decode_every", 16))
    if global_embedding is None:
        global_embedding = cache.global_embedding(voice)

    inputs = _build_inputs(cache, text, emotion)

    streamer = _make_token_streamer()
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(gcfg.get("max_new_tokens", 1024)),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        eos_token_id=list(eos_ids),
        pad_token_id=int(mcfg.get("pad_token_id", 151643)),
        streamer=streamer,
    )

    thread = threading.Thread(target=_safe_generate, args=(cache.model, gen_kwargs), daemon=True)
    thread.start()

    audio_tokens: list[int] = []
    sent = 0
    while True:
        tok = streamer.q.get()
        if tok is None:
            break
        if tok in eos_ids:
            break
        if offset <= tok < offset + vsize:
            audio_tokens.append(tok - offset)
            if len(audio_tokens) % decode_every == 0:
                wav = _decode(cache, audio_tokens, global_embedding)
                if wav is not None and wav.shape[0] > sent:
                    yield _to_i16(wav[sent:])
                    sent = wav.shape[0]
    # final flush
    wav = _decode(cache, audio_tokens, global_embedding)
    if wav is not None and wav.shape[0] > sent:
        yield _to_i16(wav[sent:])
    thread.join(timeout=1.0)


def _safe_generate(model, kwargs) -> None:
    import torch

    try:
        with torch.no_grad():
            model.generate(**kwargs)
    except Exception as exc:
        print(f"[mio] generate error: {exc}")
        try:
            kwargs["streamer"].end()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
class MioSession:
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
        emotion = data.get("emotion") or None
        voice = data.get("voice") or self.cache.config["mio"].get("default_voice")
        ref_b64 = data.get("ref_audio_b64") or None
        temperature = float(data.get("temperature", 0.9))
        top_p = float(data.get("top_p", 0.9))

        if not text:
            await self._out_q.put({"type": "error", "req_id": req_id, "message": "empty text"})
            return

        await self._out_q.put({"type": "begin", "req_id": req_id,
                               "sample_rate": self.cache.sample_rate,
                               "server_ts": time.time() * 1000.0})
        await asyncio.to_thread(self._run_synth, req_id, text, emotion, voice, ref_b64, temperature, top_p)

    def _run_synth(self, req_id, text, emotion, voice, ref_b64, temperature, top_p) -> None:
        import base64

        t_start = time.perf_counter()
        index = 0
        fcl_ms = None
        total_samples = 0
        try:
            # zero-shot clone from an uploaded reference clip, if provided
            global_embedding = None
            if ref_b64 and self.cache.config["mio"].get("allow_reference_upload", True):
                try:
                    global_embedding = self.cache.embedding_from_wav_bytes(base64.b64decode(ref_b64))
                except Exception as exc:
                    self._emit({"type": "error", "req_id": req_id,
                                "message": f"reference encode failed: {exc}"})
                    return

            for chunk in synth_blocking(self.cache, text, emotion, voice,
                                        global_embedding, temperature, top_p):
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
