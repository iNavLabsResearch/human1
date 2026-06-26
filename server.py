"""
server.py
=========
FastAPI server hosting one or both models, selected by ``enabled`` flags in
config.json:

  * human1 (Moshi, full-duplex voice)  -> WS /ws
  * veena  (Llama+SNAC, streaming TTS)  -> WS /veena

On startup it loads every *enabled* model into its static memory cache, then
opens an ngrok tunnel to port 5050 and prints the public URL.

Only a permissive allow-all CORS layer is installed (no other middleware).

Run:  python3 server.py
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
CONFIG: dict[str, Any] = json.loads((HERE / "config.json").read_text())

HUMAN1_ON = CONFIG.get("human1", {}).get("enabled", False)
VEENA_ON = CONFIG.get("veena", {}).get("enabled", False)

_H1_SEM = asyncio.Semaphore(int(CONFIG["server"].get("max_concurrent_sessions", 1)))
_VEENA_SEM = asyncio.Semaphore(int(CONFIG.get("veena", {}).get("max_concurrent_sessions", 8)))

PUBLIC_URL: str | None = None


def _open_ngrok() -> str | None:
    ncfg = CONFIG.get("ngrok", {})
    if not ncfg.get("enabled", False):
        return None
    try:
        from pyngrok import conf, ngrok

        token = ncfg.get("authtoken") or os.environ.get("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)
        if ncfg.get("region"):
            conf.get_default().region = ncfg["region"]

        port = CONFIG["server"]["port"]
        kwargs: dict[str, Any] = {"proto": "http", "bind_tls": True}
        if ncfg.get("domain"):
            kwargs["domain"] = ncfg["domain"]
        url = ngrok.connect(port, **kwargs).public_url
        wss = url.replace("https://", "wss://")
        print("=" * 64)
        print(f" ngrok public URL : {url}")
        if HUMAN1_ON:
            print(f" Human-1 WS       : {wss}/ws")
        if VEENA_ON:
            print(f" Veena TTS WS     : {wss}/veena")
        print(f" UI               : {url}/")
        print("=" * 64)
        return url
    except Exception as exc:
        print(f"[ngrok] failed to open tunnel: {exc}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if HUMAN1_ON:
        print("[startup] loading Human-1 (Moshi) ...")
        from static_memory_cache import CACHE as H1
        await asyncio.to_thread(H1.load, CONFIG)
    if VEENA_ON:
        print("[startup] loading Veena (TTS) ...")
        from veena_cache import CACHE as VC
        await asyncio.to_thread(VC.load, CONFIG)
    if not (HUMAN1_ON or VEENA_ON):
        print("[startup] WARNING: no model enabled in config.json")

    global PUBLIC_URL
    PUBLIC_URL = _open_ngrok()
    yield
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass


app = FastAPI(title="Human-1 / Veena Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["server"].get("allow_origins", ["*"]),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    out: dict[str, Any] = {"public_url": PUBLIC_URL,
                           "models": {"human1": HUMAN1_ON, "veena": VEENA_ON}}
    if HUMAN1_ON:
        from static_memory_cache import CACHE as H1
        out["human1"] = H1.status()
    if VEENA_ON:
        from veena_cache import CACHE as VC
        out["veena"] = VC.status()
    return JSONResponse(out)


@app.get("/config")
async def get_config() -> JSONResponse:
    safe = json.loads(json.dumps(CONFIG))
    safe.get("ngrok", {}).pop("authtoken", None)
    safe["public_url"] = PUBLIC_URL
    return JSONResponse(safe)


if HUMAN1_ON:
    @app.websocket("/ws")
    async def ws_human1(websocket: WebSocket) -> None:
        from static_memory_cache import CACHE as H1
        from moshi_session import MoshiSession

        if not H1.loaded:
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "message": "model not loaded"}))
            await websocket.close()
            return
        if _H1_SEM.locked():
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "message": "server busy"}))
            await websocket.close()
            return
        async with _H1_SEM:
            await MoshiSession(websocket, H1, CONFIG).run()


if VEENA_ON:
    @app.websocket("/veena")
    async def ws_veena(websocket: WebSocket) -> None:
        from veena_cache import CACHE as VC
        from veena_session import VeenaSession

        if not VC.loaded:
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "message": "veena not loaded"}))
            await websocket.close()
            return
        if _VEENA_SEM.locked():
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "message": "server busy: max sessions"}))
            await websocket.close()
            return
        async with _VEENA_SEM:
            await VeenaSession(websocket, VC).run()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((HERE / "static" / "index.html").read_text())


app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=CONFIG["server"]["host"],
        port=CONFIG["server"]["port"],
        ws_max_size=16 * 1024 * 1024,
        log_level="info",
    )
