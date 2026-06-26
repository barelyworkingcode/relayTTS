#!/usr/bin/env python3
"""relayTTS ↔ Relay enhanced-service bridge (the "service inspector" integration).

When relay spawns the daemon it injects RELAY_BRIDGE_SOCKET / RELAY_SERVICE_ID /
RELAY_SERVICE_TOKEN. Their presence means *enhanced mode*: the daemon binds a
private internal Unix socket, serves a tiny HTTP `/api/status` on it, and
registers a manifest over the bridge so relay's settings UI can show status and
edit the custom-voice palette (voices.json). Absent those vars (a standalone run
or a unit test) every entry point here is a no-op.

This is a Python port of the contract relayLLM gets for free from relay's Go
`bridge` package (see relay/cmd/testservice/main.go, relay/bridge/*.go,
relay/docs/service-manifest.md):

  * Wire: newline-delimited JSON over the bridge Unix socket. Send one
    BridgeRequest{type:"RegisterManifest", arguments:RegisterManifestRequest,
    token}; read one BridgeResponse line; type=="Error" means failure.
  * Liveness: relay tracks a manifest by the SERVICE PROCESS, not this
    connection (service_registry Forgets on exit), so we register once and
    close — no need to hold the bridge socket open.
  * Internal socket: relay polls status.path and dispatches front-door routes to
    it, sending `Authorization: Bearer <internalToken>`. The token + socket are
    service-chosen and declared in the registration.

The Eve-facing TTS protocol (length-prefixed JSON on TCP 9997) is unchanged and
unrelated — this module only adds the relay control-plane surface.
"""
import json
import os
import secrets
import shutil
import socket
import socketserver
import tempfile
import threading
from http.server import BaseHTTPRequestHandler

# Env-var ABI relay injects at spawn (relay/bridge/types.go).
ENV_BRIDGE_SOCKET = "RELAY_BRIDGE_SOCKET"
ENV_SERVICE_ID = "RELAY_SERVICE_ID"
ENV_SERVICE_TOKEN = "RELAY_SERVICE_TOKEN"
ENV_SERVICE_TOKEN_LEGACY = "RELAY_MCP_TOKEN"  # transition fallback

_MAX_LINE = 10 * 1024 * 1024  # mirrors bridge.MaxMessageSize


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


def build_clone_schema() -> list:
    """The ConfigDecl schema for cloned voices (the `clones` array).

    A clone voice = a reference recording (`ref_audio`, a 24 kHz WAV path) + its
    transcript (`ref_text`); the Base model reproduces that speaker. Reference
    audio comes in by file path because Relay's inspector edits config text and
    can't take uploads. Delivery comes from the sample's prosody — `instruct`
    does not apply (so there's no instruct field here)."""
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
                 "help": "Absolute path to a 24 kHz mono WAV of the voice to clone."},
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


def build_manifest(service_id: str, config_path: str, speakers: list) -> dict:
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
            "schema": build_voice_schema(speakers) + build_clone_schema(),
        },
    }


def build_register_payload(service_id: str, manifest: dict, internal_socket: str,
                           internal_token: str, token: str) -> bytes:
    """The exact newline-terminated BridgeRequest bytes sent to the bridge."""
    req = {
        "type": "RegisterManifest",
        "arguments": {
            "serviceId": service_id,
            "manifest": manifest,
            "internalSocket": internal_socket,
            "internalToken": internal_token,
        },
        "token": token,
    }
    return json.dumps(req).encode("utf-8") + b"\n"


# ── Internal status server (Unix-socket HTTP) ─────────────────────

class _StatusHandler(BaseHTTPRequestHandler):
    """Serves GET /api/status on the internal socket, bearer-gated.

    Relay is the only reachable client (0600 socket, same uid); the bearer is
    defense in depth. Status is read-only counters — it must never call the
    model (MLX is pinned to the generation thread)."""

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
        if self.headers.get("Authorization") != "Bearer " + self.server.token:
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


# ── Bridge client / lifecycle ─────────────────────────────────────

class RelayBridge:
    """Enhanced-service registration + internal status server.

    Construct it always; check `.enabled` (true only when relay spawned us). All
    failures in `start()` raise so the caller can keep them non-fatal — the TTS
    daemon must serve Eve on 9997 even if the inspector never comes up."""

    def __init__(self, status_provider, config_path: str, speakers: list,
                 service_id: str = None):
        self.bridge_sock = os.environ.get(ENV_BRIDGE_SOCKET, "")
        self.service_id = service_id or os.environ.get(ENV_SERVICE_ID, "")
        self.token = (os.environ.get(ENV_SERVICE_TOKEN)
                      or os.environ.get(ENV_SERVICE_TOKEN_LEGACY) or "")
        self.status_provider = status_provider
        self.config_path = config_path
        self.speakers = list(speakers)
        self.internal_token = secrets.token_hex(32)
        self.internal_sock = None
        self._dir = None
        self._server = None
        self._thread = None

    @property
    def enabled(self) -> bool:
        return bool(self.bridge_sock and self.service_id and self.token)

    def start(self):
        if not self.enabled:
            raise RuntimeError("relay bridge env not present (standalone mode)")
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
        manifest = build_manifest(self.service_id, self.config_path, self.speakers)
        payload = build_register_payload(self.service_id, manifest,
                                         self.internal_sock, self.internal_token,
                                         self.token)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
            c.settimeout(5.0)
            c.connect(self.bridge_sock)
            c.sendall(payload)
            line = self._read_line(c)
        if not line:
            raise RuntimeError("bridge closed without a response")
        resp = json.loads(line)
        if resp.get("type") == "Error":
            raise RuntimeError(f"bridge error {resp.get('code')}: {resp.get('message')}")

    @staticmethod
    def _read_line(c: socket.socket) -> str:
        buf = b""
        while b"\n" not in buf and len(buf) < _MAX_LINE:
            chunk = c.recv(65536)
            if not chunk:
                break
            buf += chunk
        return buf.split(b"\n", 1)[0].decode("utf-8")

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
