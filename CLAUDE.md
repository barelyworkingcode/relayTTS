# relayTTS

Qwen3-TTS daemon — a drop-in replacement for the Kokoro TTS daemon. Same TCP
port (9997), same length-prefixed JSON protocol, so Eve (and any other Kokoro
client) needs no changes. Qwen3-TTS is instruction-driven: delivery/emotion is
shaped by an `instruct` string per voice instead of Kokoro's flat affect.

The daemon owns no model. It is a thin protocol shell: every synthesis is an
OpenAI-compatible HTTP call to a remote server that holds the weights.

## Architecture

```
Eve Backend ──TCP 9997 (len-prefixed JSON)──► relaytts_daemon.py ──HTTPS──► remote TTS server
Relay control plane ──Unix socket (bridge)──► relay_bridge.py  (status + voices.json editor)
```

- `daemon/relaytts_daemon.py` — TCP server on 9997, request handling, voice
  registry, WAV post-processing (time-stretch, gain). Single-process,
  multi-threaded client handling.
- `RemoteEngine` (same file) — the synthesis call: builds the OpenAI-compatible
  request and renders the response through the engine's pinned-TLS opener. See
  **Engine** and **Transport security** below.
- `daemon/pinned_transport.py` — fail-closed TLS with certificate pinning for
  the transport `RemoteEngine` uses. Pure functions plus one urllib opener
  builder; no state of its own.
- `daemon/relay_bridge.py` — optional "enhanced service" surface for Relay's
  settings UI (status + voices.json editor), plus the relay launch-identity
  handshake. A no-op unless Relay launched the daemon (`RELAY_LAUNCH_FD` set).
  The Eve-facing TTS protocol on 9997 is independent of this and always works.
- `daemon/daemon_wrapper.sh` — conda-env wrapper + restart-on-crash supervisor;
  the command Relay autostarts (`--idle-timeout 0`).

## Relay launch identity

No relay credential is ever in the daemon's environment or argv (any
same-user process can read both on macOS). Relay launches the service with
`RELAY_BRIDGE_SOCKET`, `RELAY_SERVICE_ID` and `RELAY_LAUNCH_FD=3`; fd 3 is a
pipe holding a single-use 64-lowercase-hex launch secret. The contract is
`../spec-launch-identity.md`.

- First thing in `main()`, before config loading or anything that can spawn,
  `establish_launch_identity()` removes `RELAY_LAUNCH_FD` from `os.environ`,
  drains and closes the fd, validates the secret, and sends
  `{"type":"Hello","name":<service id>,"token":<secret>}` on the bridge.
  Relay answers `{"type":"OK","data":{"service_id","relay_pid"}}` and binds
  this process's kernel audit token as the service identity.
- Any failure while `RELAY_LAUNCH_FD` is set exits 78. The wrapper does not
  respawn on 78: the secret is spent, so a respawn could never re-bind.
- Every later bridge request (`RegisterManifest`) carries no `token`; relay
  authenticates it by peer audit token. Only the python process holding the
  identity may talk to the bridge — not the wrapper, not ffmpeg.
- The internal bearer for relay → daemon status calls is generated in-process
  and handed over in `RegisterManifest`; it never touches the environment.
- `RELAY_LAUNCH_FD` unset = standalone: no Hello, no bridge, inspector off.
- `config.yaml` — dev-owned static config: engine params, built-in voices,
  per-voice `instruct`, legacy Kokoro alias map. Tracked in git.
- `voices.json` — runtime, user-defined custom voices and clones. Edited live by
  Relay's settings UI; seeded empty by the daemon. **Gitignored** (it holds
  local absolute paths to reference audio), so it is not part of the source.

## Config: two sources

- **Built-in (preset) voices** + engine params + Kokoro `voice_aliases` come from
  `config.yaml`, read once at startup. `speaker` is the model's native speaker id;
  `instruct` is that voice's default delivery.
- **Custom voices** (`kind="preset"` overrides) and **clones** come from
  `voices.json`. The registry is held as a single dict snapshot, swapped
  atomically on reload, so `resolve()` / `public_voices()` never see a
  half-updated index — no lock. A bad row is skipped, never fatal.
- A `voices.json` change is hot-reloaded (mtime watch) without a daemon restart.

Voice resolution: request `voice` → exact id → alias → speaker name →
`default_voice`. Unknown ids never error; they fall back to the default.

## Voice cloning

`kind="clone"` voices in `voices.json` carry `ref_audio` + `ref_text` and render
through the remote server's Base checkpoint (`engine.remote.clone_model`).

`ref_audio` must resolve inside `clone_audio_dir` (config.yaml, default
`clone-audio` beside it) — `Config.clone_audio_path()` rejects anything outside,
including via a symlink. voices.json is edited live through relay's inspector,
so without this confinement that surface could be used to read (and exfiltrate,
since it ships upstream on the next clone request) an arbitrary file on the box.

## Engine

The daemon loads nothing — no model, no weights, no generation thread. Every
synthesis is an OpenAI-compatible `POST {base_url}/audio/speech`. Measured: 53
MB RSS.

Voice resolution, the instruct/speed/gain precedence, the ffmpeg time-stretch
and the gain clip all still run locally in the daemon, so what changes upstream
is only the model call itself.

Two things are pointedly *not* sent upstream: `speed` (the remote server's
`speed=` is treated as a no-op, and our stretch is pitch-preserving) and `gain`
(no server has it). Sending them would double-apply.

- `base_url` and `model` are required — `Config` raises at startup if either is
  missing, rather than starting clean and 400ing on every request.
- `model` is the id the **remote** server exposes, which need not match any
  local name; a router may prefix or alias its upstreams.
- Clone voices ship `ref_audio` as base64 with the request, because the
  reference recording lives beside the daemon and the server cannot see that
  path. Servers cap this (~60s of audio).
- `RELAYTTS_REMOTE_URL` / `RELAYTTS_REMOTE_MODEL` / `RELAYTTS_REMOTE_CLONE_MODEL`
  override `base_url` / `model` / `clone_model`. All three are deployment
  facts, so `config.yaml` ships them empty and a deployment never edits
  tracked config.
- A bad endpoint surfaces per-request, not at startup: the daemon still comes
  up and still serves `list_voices` if the remote host is booting behind it.

**Prefer a router over the inference server directly.** A router can hold the
upstream credential, so no token lives beside the daemon, and it is often the
only address already reachable from a VM. A plain reverse proxy forwards
`/v1/audio/speech` unchanged — the body is JSON with a `model` field, which is
all most routers need to dispatch. (`/v1/audio/transcriptions` is multipart and
may not survive a router that expects JSON.)

## Transport security

`RemoteEngine` is fail-closed TLS with optional certificate pinning
(`daemon/pinned_transport.py`), enforced by `assert_transport_config()` at
construction:

- `http` is allowed only to a loopback host (`127.0.0.1` / `::1` / `localhost`).
  Any other host must use `https`.
- `RELAYTTS_REMOTE_CA` (config: `engine.remote.ca_file`) is a PEM bundle. When
  set it is the **only** trust anchor — system roots are not consulted
  alongside it. Relay's local CA at `~/Library/Application Support/Relay/ca.crt`
  is the natural choice for relay-issued certificates.
- `RELAYTTS_REMOTE_PIN_SHA256` (config: `engine.remote.pin_sha256`) is a
  comma-separated list of SHA-256 fingerprints (hex, colons optional,
  case-insensitive) of the DER-encoded leaf certificate. When set, the
  connection fails unless the presented leaf matches one — this is what
  catches a certificate that is validly signed by a trusted CA but is not the
  one the operator expects (a compromised or substituted intermediate). Get a
  fingerprint with:

  ```
  openssl s_client -connect host:port </dev/null 2>/dev/null | openssl x509 -fingerprint -sha256 -noout
  ```
- `ca_file` / `pin_sha256` set together with an `http` URL is an error (they
  cannot apply; leaving them set would be a false sense of safety).
- There is no skip-verify / insecure flag, and none should be added.

## Protocol (TCP 9997)

4-byte big-endian length prefix + JSON body. Single, batch, batch-streaming, and
`list_voices`:

- Single: `{ "text": "...", "voice": "anna", "speed": 1.0, "instruct": "...", "gain": 1.0 }`
  → `{ "success": true, "audio_base64": "<WAV>", "sample_rate": 24000, "duration", "generation_time", "rtf" }`
- `speed`/`instruct`/`gain` omitted (not `1.0`) → use the resolved voice's own
  defaults. The remote server's `speed=` is treated as a no-op, so speed≠1.0
  is applied as a pitch-preserving ffmpeg time-stretch.
- Batch: `{ "batch": [ {item}, ... ] }`; add `"stream": true` for
  newline-delimited per-item chunks ending in a `{"type":"complete"}` line.
- `{ "action": "list_voices" }` → kokoro-shaped `{id, name, lang, gender}` list.

## Setup & run

```bash
./setup_env.sh   # conda env `relaytts` (python 3.11) + deps; requires ffmpeg (brew install ffmpeg)
RELAYTTS_REMOTE_URL=https://<host>:<port>/v1 \
RELAYTTS_REMOTE_MODEL=<id-that-server-exposes> ./build.sh
```

`build.sh` requires `RELAYTTS_REMOTE_URL` and `RELAYTTS_REMOTE_MODEL` when
registering for the first time — there is no local fallback to start without
them. `RELAYTTS_REMOTE_CLONE_MODEL`, `RELAYTTS_REMOTE_CA` and
`RELAYTTS_REMOTE_PIN_SHA256` are baked into the registration when set.

`relay service` has no update verb, so changing any of these later means
`relay service unregister --name relaytts-daemon` and re-running build.sh.

Manual run (no Relay): `python daemon/relaytts_daemon.py [--port 9997] [--idle-timeout 0]`.

CLI flags: `--host` (localhost), `--port` (9997), `--config` (or `RELAYTTS_CONFIG`),
`--custom-voices` (or `RELAYTTS_VOICES`), `--idle-timeout` seconds (0 = never
shut down; the Relay default).

Env: `RELAYTTS_REMOTE_URL`, `RELAYTTS_REMOTE_MODEL`, `RELAYTTS_REMOTE_CLONE_MODEL`,
`RELAYTTS_REMOTE_CA`, `RELAYTTS_REMOTE_PIN_SHA256`, `RELAYTTS_REMOTE_API_KEY`
(or whatever `engine.remote.api_key_env` names).

## Dependencies & supply chain

`requirements.in` is intent; `requirements.txt` is the hash-pinned lockfile
(`pip-compile --generate-hashes`). **Compile with Python 3.11**, the version
`setup_env.sh` builds the env with — pip-compile resolves against whatever
interpreter runs it, and a newer one pins wheels 3.11 cannot install.
`setup_env.sh` installs with `--require-hashes` and fails closed on any hash
mismatch. To change a dep: edit `requirements.in`, recompile, re-run the tests,
then commit both files. No espeak-ng / misaki / phonemizer / spaCy — Qwen3-TTS
has no G2P step. HTTP is urllib from the standard library, so there is no
client dependency either.

## Tests

`daemon/test_relaytts.py` (pytest, or `python daemon/test_relaytts.py` — that
delegates to pytest when installed and otherwise runs everything that needs no
fixtures, reporting what it skipped). The remote-engine payload/error tests
fake the urllib opener, so they stay offline. The transport tests in
`pinned_transport`'s section spin up a real TLS server (a CA + two leaf certs
generated with the `openssl` CLI) and exercise real handshakes, including the
pin-mismatch case; they're skipped if `openssl` isn't on PATH.

## Key paths

Paths resolve from `__file__`, not CWD: `DEFAULT_CONFIG_PATH` is `../config.yaml`
relative to the daemon; `voices.json` defaults to next to `config.yaml`.
