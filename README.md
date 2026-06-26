# Human-1 + Veena — FastAPI Voice Server

One FastAPI + WebSocket server hosting **two** models, each toggled by an
`enabled` flag in `config.json`:

- **veena** — [`maya-research/Veena`](https://huggingface.co/maya-research/Veena)
  (Llama-3B + SNAC 24 kHz) streaming **Text-to-Speech**. *Enabled by default.*
- **human1** — [`JoshTalksAI/Human-1`](https://huggingface.co/JoshTalksAI/Human-1)
  (Moshi) Hindi **full-duplex** voice. *Disabled by default.*

It loads each enabled model on startup, exposes port **5050** via **ngrok**, and
ships a browser UI with a live latency-breakdown table and a concurrency tester.

```
config.json              all tunables; per-model `enabled` flags
setup.sh                 reads the flags; installs deps + downloads enabled weights
static_memory_cache.py   Human-1 (Moshi) load/quantize/cache + duplex generation
moshi_session.py         Human-1 per-connection full-duplex streaming loop
veena_cache.py           Veena LLM + SNAC load/quantize/cache
veena_session.py         Veena streaming TTS + per-chunk latency breakdown
server.py                FastAPI app, /ws + /veena endpoints, ngrok tunnel
static/index.html        UI: Veena TTS (+ concurrency table) and Human-1 duplex
```

## Veena TTS

`/veena` WebSocket. Client sends `{"type":"tts","req_id":..,"text":..,"speaker":..,
"temperature":..,"top_p":..}`; server streams, per audio chunk, a JSON header
followed by Int16-LE PCM @ 24 kHz. The header carries the **latency breakdown**:

- `gen_ms` — model time since request start (cumulative)
- `gen_delta` — model time for *this* chunk
- `server_ms` — total server handling time (≈ `gen_ms`)

The UI adds **client recv ms** (wall time since the text was submitted) and shows
every chunk in a table. Speakers (`kavya`, `agastya`, `maitri`, `vinaya`) are
selectable in the UI. The **Concurrency** field opens N parallel WSS connections
with the same text and tabulates each request's per-chunk latency.

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
