# relayTTS

Qwen3-TTS daemon — a drop-in replacement for the Kokoro daemon. Same TCP port (9997), same length-prefixed JSON protocol. Eve needs no changes.

The daemon owns no model. It's a thin protocol shell: every synthesis is an OpenAI-compatible HTTP call to a remote server that holds the weights (`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit` or whatever the endpoint exposes).

## Quick Start

```bash
./setup_env.sh          # Create conda env + install deps
RELAYTTS_REMOTE_URL=https://<host>:<port>/v1 \
RELAYTTS_REMOTE_MODEL=<id-that-server-exposes> ./build.sh   # Unregisters kokoro-daemon, registers relaytts-daemon with Relay (autostart)
./test_speak.sh "Hello" # Test: generates speech and plays via afplay
```

## Architecture

```
Eve Web Chat ──WS──► Eve Backend ──TCP──► relayTTS Daemon (port 9997) ──HTTPS──► remote TTS server
                         │                       │
                    tts-service.js          relaytts_daemon.py
                    (Node.js TCP client)    (protocol server + RemoteEngine)
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
- **Clones** — `kind="clone"` voices carry `ref_audio` + `ref_text` and render through the remote server's Base checkpoint (`engine.remote.clone_model`). `ref_audio` is a path relative to `clone_audio_dir` (config.yaml, default `clone-audio` beside it), or absolute inside it — anything outside is rejected.

Edits to `voices.json` are hot-reloaded (mtime watch) — no daemon restart.

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

`speed`/`instruct`/`gain` are optional — omitted, they use the resolved voice's own defaults. The remote server's `speed=` is treated as a no-op, so any `speed != 1.0` is applied as a pitch-preserving ffmpeg time-stretch.

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

## Remote engine

The daemon loads nothing — no model, no weights, no generation thread. Every synthesis calls an OpenAI-compatible `POST {base_url}/audio/speech`. Measured footprint: **53 MB** resident.

The voice registry, the `instruct`/`speed`/`gain` precedence, the pitch-preserving time-stretch and the TCP protocol on 9997 all run in the daemon itself — only the model call is remote.

```yaml
engine:
  remote:
    base_url: https://198.51.100.10:8080/v1          # your server, or a router in front of it
    model: my-router/qwen3-tts-customvoice
    clone_model: my-router/qwen3-tts-base
    timeout: 120
    api_key_env: RELAYTTS_REMOTE_API_KEY   # name of the env var, never the token
    ca_file: ""            # PEM bundle; the only trust anchor when set
    pin_sha256: ""          # comma-separated leaf-certificate SHA-256 fingerprints
```

```bash
./setup_env.sh
RELAYTTS_REMOTE_URL=https://<host>:<port>/v1 \
RELAYTTS_REMOTE_MODEL=<id-that-server-exposes> ./build.sh
```

`RELAYTTS_REMOTE_MODEL` is required — `build.sh` refuses to register the service without it, and `Config` refuses to start the daemon without it either. `RELAYTTS_REMOTE_URL` is required too *unless* relay is installed, in which case `build.sh` defaults it to relay's own model socket (see "Remote via relay's model socket" below). `build.sh` bakes the resolved values (plus `RELAYTTS_REMOTE_CLONE_MODEL`, `RELAYTTS_REMOTE_CA`, `RELAYTTS_REMOTE_PIN_SHA256` when set) into the service registration, so a host's address never has to enter the config file.

Notes:

- `model` is the id the **remote** server exposes, which need not match any local name. A router may prefix its upstreams.
- If that server sits behind an LLM router you already run, point `base_url` at the router rather than the server: it can hold the upstream credential so no token has to live beside the daemon.
- Clone voices send `ref_audio` as base64 with each request — the reference recording lives beside the daemon, not on the server. Servers cap this at roughly 60 seconds of audio.
- A bad endpoint surfaces per request, not at startup, so the daemon still comes up and still answers `list_voices` if the remote host is still booting.

### Transport security

Fail-closed TLS with optional certificate pinning — see `daemon/pinned_transport.py`.

- `http` is allowed only to a loopback host (`127.0.0.1` / `::1` / `localhost`); anything else must use `https`.
- `ca_file` (`RELAYTTS_REMOTE_CA`), when set, is the **only** trust anchor — system roots are not consulted alongside it. Relay's local CA at `~/Library/Application Support/Relay/ca.crt` is the natural choice for relay-issued certificates.
- `pin_sha256` (`RELAYTTS_REMOTE_PIN_SHA256`) additionally pins the leaf certificate's SHA-256 fingerprint(s) — a certificate that is validly signed by a trusted CA but isn't the expected leaf is rejected. Get a fingerprint with:

  ```
  openssl s_client -connect host:port </dev/null 2>/dev/null | openssl x509 -fingerprint -sha256 -noout
  ```
- There is no skip-verify / insecure flag.

### Remote via relay's model socket

`RELAYTTS_REMOTE_URL` also accepts `unix:<absolute path>` — HTTP over
`AF_UNIX` against relay's own model endpoint (`relay/docs/model-endpoint.md`)
instead of a directly-configured server. `build.sh` defaults to this
(`unix:$HOME/Library/Application Support/relay/model.sock`) whenever relay is
installed and no `RELAYTTS_REMOTE_URL` is given.

- The base path is always `/v1` — the socket path names the socket file, not
  a URL prefix.
- **No `Authorization` or `x-api-key` header is ever sent on this path.**
  Relay identifies the daemon by its launch identity (the kernel's audit
  token on the connection), not by a header — a header would be judged as a
  bearer credential instead and can only make the call worse. A configured
  `RELAYTTS_REMOTE_API_KEY` (or whatever `api_key_env` names) is ignored,
  with a one-line startup warning naming the variable, never its value.
- `RELAYTTS_REMOTE_CA` / `RELAYTTS_REMOTE_PIN_SHA256` don't apply here —
  there is no TLS layer on `AF_UNIX` — and `assert_transport_config` refuses
  startup if either is set alongside a `unix:` URL, the same "would be a
  false sense of safety" rule plain `http` already gets.
- The path after `unix:` must be absolute; a relative one is refused at
  startup with a clear message.
- One connection is opened per daemon process and reused (HTTP keep-alive)
  across requests rather than one dialed and torn down per synthesis.
- relay's model endpoint returns 401 if this service doesn't hold the
  `models` capability, and the same 404 for a model that's unknown as for
  one that isn't in `--allowed-model`; 429 (admission timeout) and 503 (no
  model host registered) are retried with backoff.
- **Registering the `models` capability is presence-gated** — `build.sh`'s
  `relay service register` call raises a real macOS confirmation dialog and
  must be run at the console, not over SSH (see relayHarness's CLAUDE.md for
  the general shape of this gate). If the service is already registered,
  `build.sh` prints the exact `unregister`-then-`register` command needed to
  pick up the capability change; there is no in-place update.

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
    ├── relaytts_daemon.py     # TCP server, voice registry, RemoteEngine, WAV post-processing
    ├── pinned_transport.py    # Fail-closed TLS + certificate pinning for RemoteEngine
    ├── relay_bridge.py        # Relay launch-identity Hello + settings-UI bridge (status + voices.json editor)
    ├── daemon_wrapper.sh      # Conda wrapper + restart-on-crash supervisor
    └── test_relaytts.py       # pytest suite
```

## Built on

This daemon is a thin protocol shell — it reimplements no speech synthesis; it calls a server that does. The hard parts belong to other people:

| | |
|---|---|
| [Qwen3-TTS](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit) — Alibaba (Apache-2.0) | the model, and the instruction-driven delivery this daemon is built around |
| [Kokoro](https://github.com/hexgrad/kokoro) — hexgrad (Apache-2.0) | the TTS daemon this replaced; its port and wire protocol are kept for compatibility, which is why old voice ids still resolve |
| [soundfile](https://github.com/bastibe/python-soundfile) (BSD-3) · [NumPy](https://numpy.org) (BSD-3) · [PyYAML](https://pyyaml.org) (MIT) · [FFmpeg](https://ffmpeg.org) | encoding, arrays, config, and the pitch-preserving time-stretch |

What this repo adds is the daemon: a length-prefixed TCP protocol, a voice registry with hot reload, delivery/gain/speed handling, and the pinned-TLS transport to wherever inference actually happens.

## Dependencies

- macOS (the daemon runs as a Relay-managed service)
- Conda (miniconda/miniforge)
- ffmpeg (`brew install ffmpeg`) — runtime system dep for pitch-preserving time-stretch
- Python packages: `soundfile`, `numpy`, `pyyaml`

No espeak-ng, misaki, phonemizer, or spaCy — Qwen3-TTS needs no G2P phonemizer step. No MLX or mlx-audio — the daemon owns no model. HTTP is urllib from the standard library, so there's no client dependency either.

### Dependency management

`setup_env.sh` installs from `requirements.txt` with `--require-hashes` (fails closed on any hash mismatch); it falls back to loose install from `requirements.in` only if no lockfile exists yet. To change a dependency:

```bash
# 1. edit requirements.in
pip-compile --generate-hashes --allow-unsafe --output-file requirements.txt requirements.in
# 2. re-run the tests, then commit requirements.in + requirements.txt
```
