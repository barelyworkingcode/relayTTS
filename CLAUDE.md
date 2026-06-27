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

Manual run (no Relay): `python daemon/relaytts_daemon.py [--port 9997] [--idle-timeout 0]`.

CLI flags: `--host` (localhost), `--port` (9997), `--config` (or `RELAYTTS_CONFIG`),
`--custom-voices` (or `RELAYTTS_VOICES`), `--idle-timeout` seconds (0 = never
shut down; the Relay default).

## Dependencies & supply chain

`requirements.in` is intent; `requirements.txt` is the hash-pinned lockfile
(`pip-compile --generate-hashes`). `setup_env.sh` installs with
`--require-hashes` and fails closed on any hash mismatch. To change a dep: edit
`requirements.in`, recompile, **re-run the concurrency smoke test**, then commit
both files. No espeak-ng / misaki / phonemizer / spaCy — Qwen3-TTS has no G2P step.

## Tests

`daemon/test_relaytts.py` (pytest). Includes the concurrency smoke test that
guards the single-generation-thread invariant — run it after any dependency bump
or change to the generation path.

## Key paths

Paths resolve from `__file__`, not CWD: `DEFAULT_CONFIG_PATH` is `../config.yaml`
relative to the daemon; `voices.json` defaults to next to `config.yaml`.
