# relayTTS

Qwen3-TTS daemon — a drop-in replacement for the Kokoro daemon. Same TCP port (9997), same length-prefixed JSON protocol. Eve needs no changes.

Model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit` via mlx-audio on Apple Silicon.

## Quick Start

```bash
./setup_env.sh          # Create conda env + install deps
./build.sh              # Unregisters kokoro-daemon, registers relaytts-daemon with Relay (autostart)
./test_speak.sh "Hello" # Test: generates speech and plays via afplay
```

## Architecture

```
Eve Web Chat ──WS──► Eve Backend ──TCP──► relayTTS Daemon (port 9997)
                         │                       │
                    tts-service.js          relaytts_daemon.py
                    (Node.js TCP client)    (MLX/Qwen3-TTS server)
                         │
                    base64 WAV ──WS──► Browser (Web Audio API)
```

The daemon runs as a Relay autostart service. Eve connects via TCP, sends text + voice ID, receives base64-encoded WAV audio.

Optionally the daemon holds no model at all and calls out to a server that does — see [Remote inference](#remote-inference).

Qwen3-TTS is an instruction-driven model: each voice has an `instruct` field in `config.yaml` that shapes its emotion and delivery style (e.g. "Confident, clear and friendly."). The daemon injects the instruct when generating — no phonemizer or G2P step required.

## Voices

Built-in voices from Qwen3-TTS CustomVoice (defined in `config.yaml`):

| ID         | Name     | Lang    | Gender | Speaker    |
|------------|----------|---------|--------|------------|
| `anna`     | Anna     | English | F      | `ono_anna` (default) |
| `ryan`     | Ryan     | English | M      | `ryan`     |
| `aiden`    | Aiden    | English | M      | `aiden`    |
| `sohee`    | Sohee    | English | F      | `sohee`    |
| `serena`   | Serena   | Chinese | F      | `serena`   |
| `vivian`   | Vivian   | Chinese | F      | `vivian`   |
| `uncle_fu` | Uncle Fu | Chinese | M      | `uncle_fu` |
| `eric`     | Eric     | Chinese | M      | `eric`     |
| `dylan`    | Dylan    | Chinese | M      | `dylan`    |

Each entry specifies the `speaker` (the Qwen3 CustomVoice timbre that renders it), an `instruct` string for emotion/delivery, and `lang`/`gender`. `voice_aliases` in `config.yaml` maps legacy Kokoro IDs (e.g. `af_heart`, `am_adam`) to relayTTS voices, so existing clients need no changes. Any unknown id falls back to `default_voice` (`anna`), so old voice preferences never error.

```yaml
# config.yaml voice schema (flow style, one voice per line)
voices:
  - {id: ryan, name: Ryan, lang: English, gender: M, speaker: ryan, instruct: "Confident, clear and friendly."}

voice_aliases:
  af_heart: anna
  am_adam: ryan
```

### Custom voices and cloning (`voices.json`)

Beyond the built-in palette, users add their own voices in `voices.json` (runtime state — gitignored, seeded empty by the daemon, edited live by Relay's settings UI):

- **Custom presets** — reuse a built-in `base_speaker` with a different `instruct`/`gain`/`speed` (e.g. a "Storyteller" built on `aiden`).
- **Clones** — `kind="clone"` voices carry `ref_audio` + `ref_text` and render through a separate Qwen3 Base checkpoint (`engine.clone_repo_id`), loaded lazily on the first clone request so its RAM is only paid when cloning is used.

Edits to `voices.json` are hot-reloaded (mtime watch) — no daemon restart, no model reload.

## test_speak.sh

```bash
./test_speak.sh "Hello world"              # default voice (anna)
./test_speak.sh "Good morning" serena      # specific voice
./test_speak.sh "Fast speech" ryan 1.5     # voice + speed
./test_speak.sh --voices                   # list all voices
PORT=9998 ./test_speak.sh "test"           # alternate port
```

## Protocol

TCP on port 9997. Length-prefixed JSON (4-byte big-endian header). Identical to the Kokoro protocol.

**Request (single):**
```json
{ "text": "Hello world", "voice": "serena", "speed": 1.0, "instruct": "Bright and upbeat.", "gain": 1.0 }
```

`speed`/`instruct`/`gain` are optional — omitted, they use the resolved voice's own defaults. Qwen3's native `speed=` is a no-op, so any `speed != 1.0` is applied as a pitch-preserving ffmpeg time-stretch.

**Response:**
```json
{
  "success": true,
  "audio_base64": "<base64 WAV>",
  "sample_rate": 24000,
  "duration": 1.5,
  "generation_time": 0.4,
  "rtf": 0.27
}
```

**Other actions:** `{ "action": "list_voices" }` returns the kokoro-shaped `{id, name, lang, gender}` list. `{ "batch": [ {item}, ... ] }` synthesizes many at once; add `"stream": true` for newline-delimited per-item chunks ending in a `{"type":"complete"}` line.

## Remote inference

By default the daemon owns its model. Set `engine.remote.enabled: true` in `config.yaml` and it loads **nothing** — no MLX, no weights, no generation thread — and synthesizes by calling an OpenAI-compatible `POST {base_url}/audio/speech` instead. Measured footprint: **53 MB** resident, against 2.8 GB with the model loaded.

This is for running the daemon on a machine too small to hold the weights — a VM, a spare box — while something with a GPU does the work. Nothing else changes: the voice registry, the `instruct`/`speed`/`gain` precedence, the pitch-preserving time-stretch and the TCP protocol on 9997 are all identical, so clients cannot tell which engine is serving them.

```yaml
engine:
  remote:
    enabled: true
    base_url: http://198.51.100.10:8080/v1          # your server, or a router in front of it
    model: my-router/qwen3-tts-customvoice
    clone_model: my-router/qwen3-tts-base
    timeout: 120
    api_key_env: RELAYTTS_REMOTE_API_KEY   # name of the env var, never the token
```

Set it up with the dependency set that matches — `--remote` skips MLX entirely, roughly 2 GB lighter, and leaves the box unable to quietly fall back to loading a model:

```bash
./setup_env.sh --remote
RELAYTTS_REMOTE_URL=http://<router>:<port>/v1 ./build.sh
```

`RELAYTTS_REMOTE_URL` sets `base_url` and enables remote mode on its own, and `build.sh` bakes it into the service registration — so a host's address never has to enter the config file.

Notes:

- `model` is the id the **remote** server exposes, which is not `engine.repo_id`. A router may prefix its upstreams.
- If that server sits behind an LLM router you already run, point `base_url` at the router rather than the server: it can hold the upstream credential so no token has to live beside the daemon.
- Clone voices send `ref_audio` as base64 with each request — the reference recording lives beside the daemon, not on the server. Servers cap this at roughly 60 seconds of audio.
- A bad endpoint surfaces per request, not at startup, so the daemon still comes up and still answers `list_voices` if the remote host is still booting.

## Files

```
relayTTS/
├── README.md
├── CLAUDE.md                  # Architecture / setup notes for Claude Code
├── LICENSE                    # MIT
├── build.sh                   # Unregisters kokoro-daemon, registers relaytts-daemon with Relay
├── setup_env.sh               # Creates conda env (relaytts, python 3.11) + installs deps
├── requirements.in            # Top-level deps (intent); edit to change a dep
├── requirements.txt           # Hash-pinned lockfile (generate with pip-compile; do not hand-edit)
├── requirements-remote.in     # Same, for --remote: no MLX, no weights
├── requirements-remote.txt    # Hash-pinned lockfile for --remote
├── config.yaml                # Built-in voices, instruct strings, voice_aliases, engine params
├── voices.json                # Runtime custom voices + clones (gitignored, seeded empty)
├── test_speak.sh              # CLI test tool
└── daemon/
    ├── relaytts_daemon.py     # TCP server, MLX/Qwen3-TTS model or remote engine, WAV generation
    ├── relay_bridge.py        # Relay settings-UI bridge (status + voices.json editor)
    ├── daemon_wrapper.sh      # Conda wrapper + restart-on-crash supervisor
    └── test_relaytts.py       # pytest suite (incl. concurrency smoke test)
```

## Dependencies

- macOS with Apple Silicon
- Conda (miniconda/miniforge)
- ffmpeg (`brew install ffmpeg`) — runtime system dep for pitch-preserving time-stretch
- Python packages: `mlx-audio>=0.4.4`, `mlx`, `soundfile`, `numpy`, `pyyaml`

No espeak-ng, misaki, phonemizer, or spaCy — Qwen3-TTS needs no G2P phonemizer step.

### Dependency management

`setup_env.sh` installs from `requirements.txt` with `--require-hashes` (fails closed on any hash mismatch); it falls back to loose install from `requirements.in` only if no lockfile exists yet. To change a dependency:

```bash
# 1. edit requirements.in
pip-compile --generate-hashes --allow-unsafe --output-file requirements.txt requirements.in
# 2. re-run the concurrency smoke test, then commit requirements.in + requirements.txt
```
