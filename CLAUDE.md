# relayTTS

Qwen3-TTS daemon — a drop-in replacement for the Kokoro TTS daemon. Same TCP
port (9997), same length-prefixed JSON protocol, so Eve (and any other Kokoro
client) needs no changes. Qwen3-TTS is instruction-driven: delivery/emotion is
shaped by an `instruct` string per voice instead of Kokoro's flat affect.

Engine: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit` via mlx-audio on
Apple Silicon / MLX.

## Architecture

```
Eve Backend ──TCP 9997 (len-prefixed JSON)──► relaytts_daemon.py ──► Qwen3-TTS (MLX)
Relay control plane ──Unix socket (bridge)──► relay_bridge.py  (status + voices.json editor)
```

- `daemon/relaytts_daemon.py` — TCP server on 9997, owns the MLX model and WAV
  generation. Single-process, multi-threaded client handling.
- `RemoteEngine` — the alternative to owning a model at all. See **Remote
  engine** below.
- `daemon/relay_bridge.py` — optional "enhanced service" surface for Relay's
  settings UI (status endpoint + live voices.json editing). A no-op unless Relay
  injects `RELAY_BRIDGE_SOCKET` / `RELAY_SERVICE_ID` / `RELAY_SERVICE_TOKEN`. The
  Eve-facing TTS protocol on 9997 is independent of this and always works.
- `daemon/daemon_wrapper.sh` — conda-env wrapper + restart-on-crash supervisor;
  the command Relay autostarts (`--idle-timeout 0`).
- `config.yaml` — dev-owned static config: engine params, built-in voices,
  per-voice `instruct`, legacy Kokoro alias map. Tracked in git.
- `voices.json` — runtime, user-defined custom voices and clones. Edited live by
  Relay's settings UI; seeded empty by the daemon. **Gitignored** (it holds
  local absolute paths to reference audio), so it is not part of the source.

## Threading model (important)

Applies to the local engine only — in remote mode there is no model, no
generation thread, and none of this constrains anything.

mlx-audio's `generate()` crashes if driven from more than one thread. The daemon
loads and runs the model on **one dedicated generation thread** (`_generation_worker`);
client-handler threads enqueue work and wait for the result. Do not call the
model directly from a client handler or add a second worker. See the
`mlx-audio threading` memory.

## Config: two sources

- **Built-in (preset) voices** + engine params + Kokoro `voice_aliases` come from
  `config.yaml`, read once at startup. `speaker` is the model's native speaker id;
  `instruct` is that voice's default delivery.
- **Custom voices** (`kind="preset"` overrides) and **clones** come from
  `voices.json`. The registry is held as a single dict snapshot, swapped
  atomically on reload, so `resolve()` / `public_voices()` never see a
  half-updated index — no lock. A bad row is skipped, never fatal.
- A `voices.json` change is hot-reloaded (mtime watch) without a model reload or
  daemon restart.

Voice resolution: request `voice` → exact id → alias → speaker name →
`default_voice`. Unknown ids never error; they fall back to the default.

## Voice cloning

`kind="clone"` voices in `voices.json` carry `ref_audio` + `ref_text` and render
through a **separate** Qwen3 Base checkpoint (`engine.clone_repo_id`,
`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit`), loaded lazily on the first clone
request so its RAM is only paid when cloning is actually used.

## Remote engine

`engine.remote.enabled` (config.yaml) moves inference off this machine. The
daemon then loads **nothing** — no mlx-audio import, no weights, no generation
thread — and every synthesis becomes an OpenAI-compatible
`POST {base_url}/audio/speech`. Measured: 53 MB RSS remote vs 2.8 GB local.

Everything else is deliberately identical. Voice resolution, the
instruct/speed/gain precedence, the ffmpeg time-stretch and the gain clip all
still run locally, so a remote daemon renders a given request the same way a
local one does and clients on 9997 cannot tell which is serving them.

Two things are pointedly *not* sent upstream: `speed` (Qwen3's native `speed=`
is a no-op, and our stretch is pitch-preserving) and `gain` (no server has it).
Sending them would double-apply.

- `model` is the id the **remote** server exposes, which need not match
  `repo_id`; a router may prefix or alias its upstreams.
- Clone voices ship `ref_audio` as base64 with the request, because the
  reference recording lives beside the daemon and the server cannot see that
  path. Servers cap this (~60s of audio).
- `RELAYTTS_REMOTE_URL` overrides `base_url` **and** enables remote on its own,
  so a supervisor can flip one deployment without editing tracked config.
- A bad endpoint surfaces per-request, not at startup: the daemon still comes
  up and still serves `list_voices` if the remote host is booting behind it.

**Prefer a router over the inference server directly.** A router can hold the
upstream credential, so no token lives beside the daemon, and it is often the
only address already reachable from a VM. A plain reverse proxy forwards
`/v1/audio/speech` unchanged — the body is JSON with a `model` field, which is
all most routers need to dispatch. (`/v1/audio/transcriptions` is multipart and
may not survive a router that expects JSON.)

## Protocol (TCP 9997)

4-byte big-endian length prefix + JSON body. Single, batch, batch-streaming, and
`list_voices`:

- Single: `{ "text": "...", "voice": "anna", "speed": 1.0, "instruct": "...", "gain": 1.0 }`
  → `{ "success": true, "audio_base64": "<WAV>", "sample_rate": 24000, "duration", "generation_time", "rtf" }`
- `speed`/`instruct`/`gain` omitted (not `1.0`) → use the resolved voice's own
  defaults. Qwen3's native `speed=` is a no-op, so speed≠1.0 is applied as a
  pitch-preserving ffmpeg time-stretch.
- Batch: `{ "batch": [ {item}, ... ] }`; add `"stream": true` for
  newline-delimited per-item chunks ending in a `{"type":"complete"}` line.
- `{ "action": "list_voices" }` → kokoro-shaped `{id, name, lang, gender}` list.

## Setup & run

```bash
./setup_env.sh   # conda env `relaytts` (python 3.11) + deps; requires ffmpeg (brew install ffmpeg)
./build.sh       # unregister kokoro-daemon, register relaytts-daemon with Relay (autostart)
```

Remote deployment is the same two commands, one flag and one variable — nothing
in the source changes:

```bash
./setup_env.sh --remote                                  # no MLX, no weights
RELAYTTS_REMOTE_URL=http://<router>:<port>/v1 ./build.sh # baked into the registration
```

`--remote` installs `requirements-remote.txt` (soundfile, numpy, pyyaml) instead
of the full lock, so a remote box cannot silently fall back to loading a model —
about 2 GB lighter. Both modes build the same env name because
`daemon_wrapper.sh` activates it by name, so switching modes rebuilds the env.

`relay service` has no update verb, so changing the URL later means
`relay service unregister --name relaytts-daemon` and re-running build.sh.

Manual run (no Relay): `python daemon/relaytts_daemon.py [--port 9997] [--idle-timeout 0]`.

CLI flags: `--host` (localhost), `--port` (9997), `--config` (or `RELAYTTS_CONFIG`),
`--custom-voices` (or `RELAYTTS_VOICES`), `--idle-timeout` seconds (0 = never
shut down; the Relay default).

Env: `RELAYTTS_REMOTE_URL` (enables remote mode and sets the base URL),
`RELAYTTS_REMOTE_API_KEY` (or whatever `engine.remote.api_key_env` names).

## Dependencies & supply chain

`requirements.in` is intent; `requirements.txt` is the hash-pinned lockfile
(`pip-compile --generate-hashes`). `requirements-remote.{in,txt}` is the same
pair for `--remote` — the two resolve to different versions because the full
lock is constrained by mlx-audio and the remote one is not. `setup_env.sh`
installs with `--require-hashes` and fails closed on any hash mismatch. To change a dep: edit
`requirements.in`, recompile, **re-run the concurrency smoke test**, then commit
both files. No espeak-ng / misaki / phonemizer / spaCy — Qwen3-TTS has no G2P step.

## Tests

`daemon/test_relaytts.py` (pytest, or `python daemon/test_relaytts.py` — that
delegates to pytest when installed and otherwise runs everything that needs no
fixtures, reporting what it skipped). The remote-engine tests fake `urlopen`
and assert the request shape, so they stay offline and model-free.

Includes the concurrency smoke test that
guards the single-generation-thread invariant — run it after any dependency bump
or change to the generation path.

## Key paths

Paths resolve from `__file__`, not CWD: `DEFAULT_CONFIG_PATH` is `../config.yaml`
relative to the daemon; `voices.json` defaults to next to `config.yaml`.
