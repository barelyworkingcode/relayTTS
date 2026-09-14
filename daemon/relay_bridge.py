#!/usr/bin/env python3
"""relayTTS ↔ Relay enhanced-service bridge (the "service inspector" integration).

When relay launches the daemon it sets RELAY_BRIDGE_SOCKET / RELAY_SERVICE_ID /
RELAY_LAUNCH_FD and hands a one-shot launch secret over an inherited pipe (fd
3). No relay credential is ever in the environment. At startup the daemon
drains that pipe and presents the secret in a `Hello` on the bridge socket;
relay binds this process's kernel audit token as the service's identity, and
every later bridge request from this process carries no token at all.

With an identity bound, the daemon binds a private internal Unix socket, serves
a tiny HTTP `/api/status` on it, and registers a manifest over the bridge so
relay's settings UI can show status and edit the custom-voice palette
(voices.json). Without RELAY_LAUNCH_FD (a standalone run or a unit test) every
entry point here is a no-op.

This is a Python port of the contract relayLLM gets from relay's Go `bridge`
package (see relay/docs/service-manifest.md, relay/docs/tokens.md):

  * Wire: newline-delimited JSON over the bridge Unix socket. Send one
    BridgeRequest; read one BridgeResponse line; type=="Error" means failure.
  * Identity: bound to the process that sent Hello, not to a connection, so a
    request on any later connection from this process is authenticated.
  * Liveness: relay tracks a manifest by the SERVICE PROCESS, not this
    connection, so we register once and close.
  * Internal socket: relay polls status.path and dispatches front-door routes to
    it, sending `Authorization: Bearer <internalToken>`. The token + socket are
    service-chosen and declared in the registration.

The Eve-facing TTS protocol (length-prefixed JSON on TCP 9997) is unchanged and
unrelated — this module only adds the relay control-plane surface.
"""
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import socketserver
import tempfile
import threading
from http.server import BaseHTTPRequestHandler

# Env-var ABI relay sets at launch. None of these is secret.
ENV_BRIDGE_SOCKET = "RELAY_BRIDGE_SOCKET"
ENV_SERVICE_ID = "RELAY_SERVICE_ID"
ENV_LAUNCH_FD = "RELAY_LAUNCH_FD"

_MAX_LINE = 10 * 1024 * 1024  # mirrors bridge.MaxMessageSize
_LAUNCH_SECRET_RE = re.compile(r"[0-9a-f]{64}")
# A well-formed secret is 64 bytes; anything past this is malformed, so the
# read stops rather than buffering whatever an unexpected fd produces.
_LAUNCH_SECRET_READ_CAP = 4096
_HELLO_TIMEOUT_S = 5.0


class LaunchIdentityError(RuntimeError):
    """Relay launched us but the launch identity could not be established.
    Messages never contain the secret."""


def _read_line(c: socket.socket) -> str:
    buf = b""
    while b"\n" not in buf and len(buf) < _MAX_LINE:
        chunk = c.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0].decode("utf-8", "replace")


# ── Launch identity (fd 3 secret + Hello) ─────────────────────────

def read_launch_secret(fd: int) -> str:
    """Drain `fd` to EOF, close it, and return the 64-lowercase-hex secret."""
    data = b""
    try:
        while len(data) <= _LAUNCH_SECRET_READ_CAP:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            data += chunk
    except OSError as e:
        raise LaunchIdentityError(f"cannot read launch fd {fd}: {e.strerror}") from None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        secret = data.decode("ascii")
    except UnicodeDecodeError:
        secret = ""
    if not _LAUNCH_SECRET_RE.fullmatch(secret):
        raise LaunchIdentityError(
            f"launch secret malformed ({len(data)} bytes; want 64 lowercase hex)")
    return secret


def build_hello_payload(service_id: str, secret: str) -> bytes:
    return json.dumps({"type": "Hello", "name": service_id,
                       "token": secret}).encode("utf-8") + b"\n"


def send_hello(bridge_sock: str, service_id: str, secret: str,
               timeout: float = _HELLO_TIMEOUT_S) -> dict:
    """Present the launch secret; return the OK frame's `data` on success."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
            c.settimeout(timeout)
            c.connect(bridge_sock)
            c.sendall(build_hello_payload(service_id, secret))
            line = _read_line(c)
    except OSError as e:
        raise LaunchIdentityError(f"Hello transport failure: {e.__class__.__name__}") from None
    if not line:
        raise LaunchIdentityError("bridge closed without answering Hello")
    try:
        resp = json.loads(line)
    except ValueError:
        resp = None
    if not isinstance(resp, dict):
        raise LaunchIdentityError("malformed Hello response")
    if resp.get("type") == "Error":
        raise LaunchIdentityError(
            f"Hello refused: code {resp.get('code')}: {resp.get('message')}")
    data = resp.get("data")
    if (resp.get("type") != "OK" or not isinstance(data, dict)
            or data.get("service_id") != service_id
            or not isinstance(data.get("relay_pid"), int)):
        raise LaunchIdentityError("malformed Hello response")
    return data


def establish_launch_identity(environ=os.environ) -> bool:
    """Run the launch handshake if relay launched us.

    Returns False when RELAY_LAUNCH_FD is unset (standalone), True once relay
    has bound this process's identity, and raises LaunchIdentityError on any
    failure in between. Must run before the daemon spawns any child.
    """
    raw_fd = environ.get(ENV_LAUNCH_FD)
    if raw_fd is None:
        return False
    # Deliberate: removed before anything else can fail, so no child spawned
    # later (ffmpeg) inherits a pointer to a launch pipe.
    del environ[ENV_LAUNCH_FD]
    try:
        fd = int(raw_fd)
    except ValueError:
        fd = -1
    if fd < 0:
        raise LaunchIdentityError(f"{ENV_LAUNCH_FD} is not a file descriptor")
    secret = read_launch_secret(fd)
    bridge_sock = environ.get(ENV_BRIDGE_SOCKET, "")
    service_id = environ.get(ENV_SERVICE_ID, "")
    if not bridge_sock or not service_id:
        raise LaunchIdentityError(
            f"{ENV_LAUNCH_FD} is set but {ENV_BRIDGE_SOCKET} or {ENV_SERVICE_ID} is empty")
    send_hello(bridge_sock, service_id, secret)
    return True


# ── Manifest building (pure — unit-tested without any socket) ──────

def build_voice_schema(speakers: list) -> list:
    """The ConfigDecl schema relay renders into the voices.json editor.

    One editable `voices` array; each row is a custom voice = a built-in speaker
    (timbre) restyled by a delivery prompt, plus optional gain/speed. This is the
    "create a custom voice by prompting" surface — the `instruct` field is the
    prompt. `base_speaker` is a select over the model's built-in speakers so a row
    can never name a timbre the engine doesn't have.
    """
    return [{
        "id": "voices",
        "label": "Custom voices",
        "type": "array",
        "help": ("Voices defined here appear in Eve's voice picker. Each one "
                 "restyles a built-in speaker with a delivery prompt. Built-in "
                 "voices live in config.yaml and are not edited here."),
        "item": {
            "id": "voice",
            "type": "object",
            "fields": [
                {"id": "id", "label": "Voice ID", "type": "text", "required": True,
                 "help": "Unique id Eve stores and sends back (e.g. 'narrator')."},
                {"id": "name", "label": "Display name", "type": "text", "required": True},
                {"id": "base_speaker", "label": "Base speaker", "type": "select",
                 "options": list(speakers), "required": True,
                 "help": "The built-in Qwen3 timbre this voice speaks with."},
                {"id": "instruct", "label": "Delivery prompt", "type": "textarea",
                 "help": "Natural-language style/emotion, e.g. "
                         "'Bright, energetic young British woman, gently teasing.'"},
                {"id": "lang", "label": "Language", "type": "text", "placeholder": "English",
                 "help": "Groups the voice in Eve's picker."},
                {"id": "gender", "label": "Gender", "type": "select", "options": ["F", "M"]},
                {"id": "gain", "label": "Gain", "type": "number",
                 "help": "Loudness multiplier (1.0 unchanged, 1.4 louder, 0.6 softer)."},
                {"id": "speed", "label": "Speed", "type": "number",
                 "help": "Tempo multiplier (1.0 unchanged; pitch preserved)."},
            ],
        },
    }]


def build_clone_schema(clone_audio_dir: str) -> list:
    """The ConfigDecl schema for cloned voices (the `clones` array).

    A clone voice = a reference recording (`ref_audio`, a 24 kHz WAV path) + its
    transcript (`ref_text`); the Base model reproduces that speaker. Reference
    audio comes in by file path because Relay's inspector edits config text and
    can't take uploads. `ref_audio` must resolve inside `clone_audio_dir` (the
    daemon rejects anything else), so the help text names that directory rather
    than inviting an absolute path elsewhere. Delivery comes from the sample's
    prosody — `instruct` does not apply (so there's no instruct field here)."""
    return [{
        "id": "clones",
        "label": "Cloned voices",
        "type": "array",
        "help": ("Voices cloned from a reference recording. Provide a 24 kHz WAV "
                 "and its exact transcript. Uses the Qwen3 Base model, loaded on "
                 "first use. Delivery follows the sample's prosody (no instruct)."),
        "item": {
            "id": "clone",
            "type": "object",
            "fields": [
                {"id": "id", "label": "Voice ID", "type": "text", "required": True,
                 "help": "Unique id Eve stores and sends back (e.g. 'my_voice')."},
                {"id": "name", "label": "Display name", "type": "text", "required": True},
                {"id": "ref_audio", "label": "Reference audio (path)", "type": "text",
                 "required": True,
                 "help": f"Path to a 24 kHz mono WAV, relative to {clone_audio_dir} "
                         "(or absolute inside it)."},
                {"id": "ref_text", "label": "Reference transcript", "type": "textarea",
                 "required": True, "help": "The exact words spoken in the reference audio."},
                {"id": "lang", "label": "Language", "type": "text", "placeholder": "English",
                 "help": "Groups the voice in Eve's picker."},
                {"id": "gender", "label": "Gender", "type": "select", "options": ["F", "M"]},
                {"id": "gain", "label": "Gain", "type": "number",
                 "help": "Loudness multiplier (1.0 unchanged)."},
                {"id": "speed", "label": "Speed", "type": "number",
                 "help": "Tempo multiplier (1.0 unchanged; pitch preserved)."},
            ],
        },
    }]


def build_manifest(service_id: str, config_path: str, speakers: list,
                   clone_audio_dir: str) -> dict:
    """Assemble the manifest relay validates and stores.

    `routes` must be non-empty (relay rejects an empty list), but Eve talks to
    the daemon directly over TCP 9997, not through relay's front door — so we
    declare a single namespaced placeholder route the internal server simply
    404s. status + config are the surfaces that matter. applyMode "live" means
    relay writes voices.json without restarting us (no slow model reload); the
    daemon hot-reloads the file itself.
    """
    return {
        "routes": [f"/api/{service_id}/"],
        "status": {"path": "/api/status"},
        "config": {
            "path": config_path,
            "format": "json",
            "label": "voices.json",
            "help": ("Custom voices for relayTTS — they appear in Eve's voice "
                     "picker. Built-in voices live in config.yaml."),
            "applyMode": "live",
            "schema": build_voice_schema(speakers) + build_clone_schema(clone_audio_dir),
        },
    }


def build_register_payload(service_id: str, manifest: dict, internal_socket: str,
                           internal_token: str) -> bytes:
    """The exact newline-terminated BridgeRequest bytes sent to the bridge.

    Deliberately carries no `token`: relay authenticates this request by the
    peer audit token bound at Hello."""
    req = {
        "type": "RegisterManifest",
        "arguments": {
            "serviceId": service_id,
            "manifest": manifest,
            "internalSocket": internal_socket,
            "internalToken": internal_token,
        },
    }
    return json.dumps(req).encode("utf-8") + b"\n"


# ── Internal status server (Unix-socket HTTP) ─────────────────────

class _StatusHandler(BaseHTTPRequestHandler):
    """Serves GET /api/status on the internal socket, bearer-gated.

    Relay is the only reachable client (0600 socket, same uid); the bearer is
    defense in depth. Status is read-only counters — it must never make a
    remote synthesis call."""

    def do_GET(self):
        if not self._authorized():
            return
        if self.path.split("?", 1)[0].rstrip("/") in ("/api/status", ""):
            try:
                body = json.dumps(self.server.status_provider()).encode("utf-8")
            except Exception as e:  # never let a status hiccup kill the poll
                self._send(500, json.dumps({"error": str(e)}).encode("utf-8"))
                return
            self._send(200, body)
        else:
            self._send(404, b'{"error":"not found"}')

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization")
        if header is None or not hmac.compare_digest(
                header.encode("utf-8"), ("Bearer " + self.server.token).encode("utf-8")):
            self._send(401, b'{"error":"unauthorized"}')
            return False
        return True

    def _send(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    # Unix-socket peers have no host:port — keep BaseHTTPRequestHandler's
    # logging from indexing an empty client_address, and stay quiet.
    def address_string(self) -> str:
        return "unix"

    def log_message(self, *args):
        pass


class _UnixHTTPServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, sock_path: str, token: str, status_provider):
        self.token = token
        self.status_provider = status_provider
        super().__init__(sock_path, _StatusHandler)


def _sweep_stale_bridge_dirs():
    """Remove leftover relaytts-bridge-* temp dirs from daemons that are no
    longer running.

    Only `stop()` cleans up its own dir, so a crash or a supervisor restart
    leaks one every time. A dir's internal.sock still accepting connections is
    the one signal that its daemon is alive; anything else (no socket, or a
    connection it refuses) means it's safe to remove.
    """
    base = tempfile.gettempdir()
    try:
        names = os.listdir(base)
    except OSError:
        return
    for name in names:
        if not name.startswith("relaytts-bridge-"):
            continue
        path = os.path.join(base, name)
        sock_path = os.path.join(path, "internal.sock")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
                c.settimeout(0.2)
                c.connect(sock_path)
        except (FileNotFoundError, ConnectionRefusedError):
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass  # anything else is inconclusive; leave it rather than guess


# ── Bridge client / lifecycle ─────────────────────────────────────

class RelayBridge:
    """Enhanced-service registration + internal status server.

    Construct it always; check `.enabled` (true only once relay has bound this
    process's launch identity, see `establish_launch_identity`). All
    failures in `start()` raise so the caller can keep them non-fatal — the TTS
    daemon must serve Eve on 9997 even if the inspector never comes up."""

    def __init__(self, status_provider, config_path: str, speakers: list,
                 clone_audio_dir: str, identity_bound: bool, service_id: str = None):
        self.bridge_sock = os.environ.get(ENV_BRIDGE_SOCKET, "")
        self.service_id = service_id or os.environ.get(ENV_SERVICE_ID, "")
        self.identity_bound = identity_bound
        self.status_provider = status_provider
        self.config_path = config_path
        self.speakers = list(speakers)
        self.clone_audio_dir = clone_audio_dir
        self.internal_token = secrets.token_hex(32)
        self.internal_sock = None
        self._dir = None
        self._server = None
        self._thread = None

    @property
    def enabled(self) -> bool:
        return bool(self.bridge_sock and self.service_id and self.identity_bound)

    def start(self):
        if not self.enabled:
            raise RuntimeError("no relay launch identity (standalone mode)")
        _sweep_stale_bridge_dirs()
        self._dir = tempfile.mkdtemp(prefix="relaytts-bridge-")
        self.internal_sock = os.path.join(self._dir, "internal.sock")
        self._server = _UnixHTTPServer(self.internal_sock, self.internal_token,
                                       self.status_provider)
        os.chmod(self.internal_sock, 0o600)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="relay-bridge-status", daemon=True)
        self._thread.start()
        self._register()

    def _register(self):
        manifest = build_manifest(self.service_id, self.config_path, self.speakers,
                                  self.clone_audio_dir)
        payload = build_register_payload(self.service_id, manifest,
                                         self.internal_sock, self.internal_token)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
            c.settimeout(5.0)
            c.connect(self.bridge_sock)
            c.sendall(payload)
            line = _read_line(c)
        if not line:
            raise RuntimeError("bridge closed without a response")
        resp = json.loads(line)
        if resp.get("type") == "Error":
            raise RuntimeError(f"bridge error {resp.get('code')}: {resp.get('message')}")

    def stop(self):
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        if self._dir and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
