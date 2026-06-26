# Human-1 — FastAPI Full-Duplex Server

A FastAPI + WebSocket server for [`JoshTalksAI/Human-1`](https://huggingface.co/JoshTalksAI/Human-1),
a Moshi-based Hindi full-duplex conversational model. Loads the model **INT8** on
startup, exposes port **5050** through **ngrok**, and ships a browser UI for
real-time two-way voice.

```
config.json              all tunables (server, ngrok, model, generation, audio)
setup.sh                 installs torch/moshi/pyngrok/... and downloads weights
static_memory_cache.py   loads + quantizes + caches the model once, on startup
moshi_session.py         per-connection full-duplex streaming loop
server.py                FastAPI app, /ws endpoint, ngrok tunnel
static/index.html        UI: base URL, system prompt, voice id, stats
```

## Run (on a GPU box — Kaggle / Colab / remote H100)

```bash
chmod +x setup.sh && ./setup.sh      # installs deps + downloads ~31 GB weights
python3 server.py                    # loads model, opens ngrok, serves :5050
```

On startup the console prints the public ngrok URL, e.g.:

```
ngrok public URL : https://xxxx.ngrok-free.app
WebSocket URL    : wss://xxxx.ngrok-free.app/ws
UI               : https://xxxx.ngrok-free.app/
```

Open the UI URL. The mic streams to the model and you hear Moshi reply — full
duplex (both talk at once). Paste the same ngrok URL into **Server Base URL** if
you open the page from somewhere else.

## Endpoints

| Method | Path      | Purpose                                            |
|--------|-----------|----------------------------------------------------|
| GET    | `/`       | Browser UI                                         |
| GET    | `/health` | Load status, device, quantization, public URL      |
| GET    | `/config` | Effective config (auth token redacted)             |
| WS     | `/ws`     | Full-duplex audio + text + stats                   |

## WebSocket protocol

Client → server: `{"type":"start","system_prompt":..,"voice_id":..}`, then
binary **Int16-LE mono PCM @ 24 kHz** mic frames, plus `{"type":"ping","t":ms}`.

Server → client: binary **Int16-LE PCM @ 24 kHz** (Moshi voice),
`{"type":"text",..}`, `{"type":"stats","chunk_latency_ms":..,"rtf":..,"frames":..}`,
`{"type":"pong",..}`. The UI shows **chunk latency** and **ping** from the backend.

## Configuration (`config.json`)

- `model.quantization`: `int8` (default, torchao weight-only) · `bf16` · `fp16`
- `model.device_map`: `single` · `balanced`/`auto` (multi-GPU via accelerate)
- `generation`: `temp`, `temp_text`, `top_k`, `top_k_text`
- `ngrok`: `enabled`, `authtoken`, `domain`, `region`

## Multi-GPU (Kaggle 2× GPU)

Set `"device_map": "balanced"` in `config.json`. The LM is dispatched across all
visible GPUs with `accelerate`; Mimi stays on the primary device. INT8 keeps the
~7.5 B-param LM under ~8 GB so it fits a single T4 as well.

## Notes

- Base Moshi has no native system-prompt / voice-id channel; both are plumbed
  through the protocol and applied best-effort (stored per session).
- `max_concurrent_sessions` (default **1**) serializes the shared streaming
  state — raise it only if you instantiate independent model copies.
