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
├── config.yaml                # Built-in voices, instruct strings, voice_aliases, engine params
├── voices.json                # Runtime custom voices + clones (gitignored, seeded empty)
├── test_speak.sh              # CLI test tool
└── daemon/
    ├── relaytts_daemon.py     # TCP server, MLX/Qwen3-TTS model, WAV generation
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
