"""
server.py
=========
FastAPI server for the Human-1 (Moshi) Hindi full-duplex model.

On startup it:
  1. loads config.json (every tunable lives there),
  2. loads + quantizes the model into static_memory_cache.CACHE,
  3. opens an ngrok tunnel to port 5050 and prints the public URL.

No CORS middleware is installed beyond a permissive allow-all (per request).

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

from static_memory_cache import CACHE
from moshi_session import MoshiSession

HERE = Path(__file__).parent
CONFIG: dict[str, Any] = json.loads((HERE / "config.json").read_text())

# A semaphore caps concurrent sessions (streaming state is per-connection but
# the underlying modules are shared, so we serialize by default).
_MAX_SESSIONS = int(CONFIG["server"].get("max_concurrent_sessions", 1))
_SESSION_SEM = asyncio.Semaphore(_MAX_SESSIONS)

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
        tunnel = ngrok.connect(port, **kwargs)
        url = tunnel.public_url
        print("=" * 64)
        print(f" ngrok public URL : {url}")
        print(f" WebSocket URL    : {url.replace('https://', 'wss://')}/ws")
        print(f" UI               : {url}/")
        print("=" * 64)
        return url
    except Exception as exc:
        print(f"[ngrok] failed to open tunnel: {exc}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) load the model on startup (blocking work -> thread)
    print("[startup] loading model into static memory cache ...")
    await asyncio.to_thread(CACHE.load, CONFIG)
    # 2) expose port 5050 via ngrok
    global PUBLIC_URL
    PUBLIC_URL = _open_ngrok()
    yield
    # shutdown
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass


app = FastAPI(title="Human-1 Server", lifespan=lifespan)

# Allow all origins, no other middleware (per request).
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["server"].get("allow_origins", ["*"]),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok" if CACHE.loaded else "loading",
        "public_url": PUBLIC_URL,
        "cache": CACHE.status(),
        "generation": CONFIG["generation"],
        "quantization": CONFIG["model"]["quantization"],
    })


@app.get("/config")
async def get_config() -> JSONResponse:
    safe = json.loads(json.dumps(CONFIG))
    safe.get("ngrok", {}).pop("authtoken", None)  # never leak the token
    safe["public_url"] = PUBLIC_URL
    return JSONResponse(safe)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    if not CACHE.loaded:
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "message": "model not loaded yet"}))
        await websocket.close()
        return

    # Honor the concurrent-session cap.
    if _SESSION_SEM.locked():
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "message": "server busy: max sessions reached"}))
        await websocket.close()
        return

    async with _SESSION_SEM:
        session = MoshiSession(websocket, CACHE, CONFIG)
        await session.run()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((HERE / "static" / "index.html").read_text())


# Serve any extra static assets (none required, but handy).
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
