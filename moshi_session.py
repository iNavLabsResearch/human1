"""
moshi_session.py
================
One :class:`MoshiSession` per WebSocket connection. Runs the Moshi full-duplex
streaming loop:

    mic PCM  --> Mimi.encode --> LMGen.step --> Mimi.decode --> Moshi PCM
                                     |
                                     +--> text token --> Hindi text

Threading model (full duplex, non-blocking):
    * async `_recv_loop`   : reads the socket, pushes input PCM to a thread queue
    * worker thread        : owns the streaming state, does all torch work,
                             pushes audio/text/stats to an asyncio queue
    * async `_send_loop`   : drains the asyncio queue back to the client

Wire protocol
-------------
Client -> server:
    text  {"type":"start","system_prompt":str,"voice_id":str,"sample_rate":int}
    bin   Int16-LE mono PCM @ 24 kHz  (microphone)
    text  {"type":"ping","t":<client_ms>}
    text  {"type":"stop"}

Server -> client:
    text  {"type":"ready", ...}
    bin   Int16-LE mono PCM @ 24 kHz  (Moshi voice)
    text  {"type":"text","text":str}
    text  {"type":"stats","chunk_latency_ms":float,"rtf":float,"frames":int}
    text  {"type":"pong","t":<client_ms>,"server_t":<server_ms>}
    text  {"type":"error","message":str}
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import Any, Optional

import numpy as np

# Sentinels for the cross-thread queues.
_STOP = object()


class MoshiSession:
    def __init__(self, websocket, cache, config: dict[str, Any]) -> None:
        self.ws = websocket
        self.cache = cache
        self.config = config

        self.system_prompt: str = ""
        self.voice_id: str = "default"

        self.frame_size = cache.frame_size
        self.sample_rate = cache.sample_rate

        # cross-thread plumbing
        self._in_q: "queue.Queue[Any]" = queue.Queue()      # float32 np arrays / _STOP
        self._out_q: "asyncio.Queue[Any]" = asyncio.Queue()  # outbound messages
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker: Optional[threading.Thread] = None
        self._closed = threading.Event()

        # rolling stats
        self._frames = 0
        self._gen_ms_ewma = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        await self.ws.accept()
        self._loop = asyncio.get_running_loop()

        # 1) wait for the {"type":"start"} handshake
        try:
            start = await self._recv_start()
        except Exception as exc:
            await self._safe_send_json({"type": "error", "message": f"handshake failed: {exc}"})
            await self._safe_close()
            return

        self.system_prompt = start.get("system_prompt", "") or ""
        self.voice_id = start.get("voice_id", "default") or "default"

        await self._safe_send_json({
            "type": "ready",
            "sample_rate": self.sample_rate,
            "frame_rate": self.cache.frame_rate,
            "frame_size": self.frame_size,
            "voice_id": self.voice_id,
            "system_prompt": self.system_prompt,
        })

        # 2) start worker thread + send loop, run recv loop
        self._worker = threading.Thread(target=self._worker_main, daemon=True)
        self._worker.start()
        send_task = asyncio.create_task(self._send_loop())
        try:
            await self._recv_loop()
        finally:
            self._closed.set()
            self._in_q.put(_STOP)
            await self._out_q.put(_STOP)
            await send_task
            await self._safe_close()

    async def _recv_start(self) -> dict[str, Any]:
        # Pull messages until we see a JSON "start".
        for _ in range(50):
            msg = await self.ws.receive()
            if "text" in msg and msg["text"] is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "start":
                    return data
        raise RuntimeError("no start message received")

    # ------------------------------------------------------------------ #
    # Receive (socket -> worker)
    # ------------------------------------------------------------------ #
    async def _recv_loop(self) -> None:
        while not self._closed.is_set():
            try:
                msg = await self.ws.receive()
            except Exception:
                break

            if msg.get("type") == "websocket.disconnect":
                break

            # binary PCM from the mic
            if msg.get("bytes") is not None:
                pcm = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                if pcm.size:
                    self._in_q.put(pcm)
                continue

            # control JSON
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue
                kind = data.get("type")
                if kind == "ping":
                    await self._out_q.put({
                        "type": "pong",
                        "t": data.get("t"),
                        "server_t": time.time() * 1000.0,
                    })
                elif kind == "stop":
                    break

    # ------------------------------------------------------------------ #
    # Worker thread (all torch work lives here)
    # ------------------------------------------------------------------ #
    def _worker_main(self) -> None:
        import torch
        from static_memory_cache import config_lmgen

        try:
            lm_gen = config_lmgen(self.cache.lm, self.config)
            mimi = self.cache.mimi
            device = self.cache.device
            text_tok = self.cache.text_tokenizer

            buf = np.zeros(0, dtype=np.float32)
            stats_every = max(1, int(self.cache.frame_rate))  # ~ once per second

            with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
                while not self._closed.is_set():
                    try:
                        item = self._in_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if item is _STOP:
                        break

                    buf = np.concatenate([buf, item])
                    while buf.shape[0] >= self.frame_size and not self._closed.is_set():
                        frame = buf[: self.frame_size]
                        buf = buf[self.frame_size:]

                        t0 = time.time()
                        chunk = torch.from_numpy(frame).to(device)[None, None]
                        codes = mimi.encode(chunk)  # [1, K, T]

                        for t in range(codes.shape[-1]):
                            tokens = lm_gen.step(codes[:, :, t: t + 1])
                            if tokens is None:
                                continue  # initial acoustic delay

                            text_token = int(tokens[0, 0, 0].item())
                            if text_token not in (0, 3):
                                piece = text_tok.id_to_piece(text_token)
                                txt = piece.replace("▁", " ")
                                self._emit({"type": "text", "text": txt})

                            audio_codes = tokens[:, 1:]  # [1, K, 1]
                            pcm_out = mimi.decode(audio_codes)  # [1, 1, frame_size]
                            pcm_np = pcm_out[0, 0].clamp(-1, 1).cpu().numpy()
                            pcm_i16 = (pcm_np * 32767.0).astype("<i2").tobytes()
                            self._emit(pcm_i16)

                        gen_ms = (time.time() - t0) * 1000.0
                        # exponential moving average for a stable readout
                        a = 0.2
                        self._gen_ms_ewma = (1 - a) * self._gen_ms_ewma + a * gen_ms
                        self._frames += 1

                        if self._frames % stats_every == 0:
                            frame_budget_ms = 1000.0 / self.cache.frame_rate
                            self._emit({
                                "type": "stats",
                                "chunk_latency_ms": round(self._gen_ms_ewma, 2),
                                "rtf": round(self._gen_ms_ewma / frame_budget_ms, 3),
                                "frames": self._frames,
                            })
        except Exception as exc:  # report to the client, do not crash the server
            self._emit({"type": "error", "message": f"worker: {type(exc).__name__}: {exc}"})
        finally:
            self._emit(_STOP)

    def _emit(self, item: Any) -> None:
        """Thread-safe handoff from the worker into the asyncio out-queue."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._out_q.put_nowait, item)
        except RuntimeError:
            pass  # loop closed

    # ------------------------------------------------------------------ #
    # Send (worker -> socket)
    # ------------------------------------------------------------------ #
    async def _send_loop(self) -> None:
        while True:
            item = await self._out_q.get()
            if item is _STOP:
                break
            try:
                if isinstance(item, (bytes, bytearray)):
                    await self.ws.send_bytes(item)
                else:
                    await self.ws.send_text(json.dumps(item))
            except Exception:
                break

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    async def _safe_send_json(self, obj: dict[str, Any]) -> None:
        try:
            await self.ws.send_text(json.dumps(obj))
        except Exception:
            pass

    async def _safe_close(self) -> None:
        try:
            await self.ws.close()
        except Exception:
            pass
