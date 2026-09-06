#!/usr/bin/env python3
"""
relayTTS Daemon Server — Qwen3-TTS CustomVoice.

A drop-in replacement for the Kokoro TTS daemon: same length-prefixed JSON TCP
protocol on the same port (9997), so existing clients (Eve) need no changes.
The daemon owns no model itself — every synthesis is an OpenAI-compatible
HTTP call to a remote server that does (see RemoteEngine below). Returns
base64-encoded WAV audio in JSON responses.

Voices and per-voice delivery (`instruct`) live in config.yaml, not in code.
"""
import argparse
import base64
import io
import json
import math
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import yaml

from pinned_transport import assert_transport_config, build_opener, parse_pins

DAEMON_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(DAEMON_DIR), "config.yaml")

# Wire-request bounds. speed/gain are clamped (never rejected, see
# clamp_factor) so an old client sending an odd-but-numeric value keeps
# working; text/batch/frame size are hard caps against unbounded input.
SPEED_RANGE = (0.25, 4.0)
GAIN_RANGE = (0.0, 4.0)
MAX_TEXT_CHARS = 5000
MAX_BATCH_ITEMS = 64
MAX_FRAME_BYTES = 16 * 1024 * 1024
CLIENT_TIMEOUT_S = 60.0
LISTEN_BACKLOG = 64


def clamp_factor(value, lo: float, hi: float, name: str):
    """Clamp a wire-supplied speed/gain factor into [lo, hi], or None through.

    Clamping rather than rejecting an out-of-range value is deliberate: the
    drop-in contract is that an old client sending an odd-but-numeric value
    never errors. Only a non-numeric or non-finite value is rejected, since
    there is no sane number to clamp it to."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return min(hi, max(lo, value))


# ── Config ────────────────────────────────────────────────────────

def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Config:
    """Loads and indexes voices. Built-in voices + engine params + aliases come
    from config.yaml (dev-owned, static); user-defined *custom* voices come from
    a separate voices.json that relay's settings UI edits live. Owns the merged
    voice registry, the legacy alias map, and the defaults.

    The merged registry is held as a single dict snapshot in `self._reg` and
    swapped atomically on reload, so the many client threads that call
    `resolve()` / `public_voices()` never see a half-updated index — no lock."""

    def __init__(self, path: str, custom_voices_path: str = None):
        self.path = path
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        engine = raw.get("engine", {})
        self.sample_rate = int(engine.get("sample_rate", 24000))
        self.lang_code = engine.get("lang_code", "english")
        self.temperature = float(engine.get("temperature", 0.9))
        # The only engine: every synthesis is an HTTP call to a remote server.
        # Raises ValueError if the endpoint isn't configured — see RemoteEngine.
        self.remote = RemoteEngine(engine.get("remote"))

        self.default_voice = raw.get("default_voice", "anna")
        self.default_instruct = raw.get(
            "default_instruct", "Warm, mature and composed.")

        # Built-in voices, in declared order, tagged kind="preset".
        self._builtin = [dict(v, kind="preset") for v in raw.get("voices", [])]
        self.aliases = dict(raw.get("voice_aliases", {}))
        # The model's built-in speakers — the only timbres a custom voice may
        # name. Surfaced to relay's UI as the base_speaker select options.
        self.speakers = [v["speaker"] for v in self._builtin]

        # Confines where a `ref_audio` path in voices.json may point. voices.json
        # is relay-editable, so without this an operator (or the inspector) could
        # name any file on the box and have it read and shipped off-box on the
        # next clone request. Not created here — an absent directory just means
        # no clone voice can resolve until one exists.
        clone_audio_dir = raw.get("clone_audio_dir") or "clone-audio"
        if not os.path.isabs(clone_audio_dir):
            clone_audio_dir = os.path.join(
                os.path.dirname(os.path.abspath(path)), clone_audio_dir)
        self.clone_audio_dir = os.path.realpath(clone_audio_dir)

        # Custom voices live next to config.yaml by default. relay reads/writes
        # this file directly and never creates it, so we seed an empty one.
        self.custom_voices_path = custom_voices_path or os.path.join(
            os.path.dirname(os.path.abspath(path)), "voices.json")
        self._seed_custom_file()
        self._reg = None
        self.reload_custom()

        if self.default_voice not in self._reg["by_id"]:
            raise ValueError(
                f"default_voice {self.default_voice!r} is not a defined voice")

    # ── Custom voice palette (relay-editable, hot-reloadable) ──────

    def _seed_custom_file(self):
        if os.path.exists(self.custom_voices_path):
            return
        try:
            with open(self.custom_voices_path, "w") as f:
                json.dump({"voices": [], "clones": []}, f, indent=2)
            print(f"Seeded empty custom voices file: {self.custom_voices_path}")
        except OSError as e:
            print(f"Could not seed {self.custom_voices_path}: {e}")

    def _normalize_custom(self, entry: dict, known_speakers: set) -> dict | None:
        """Coerce one voices.json row into a render spec, or None if unusable.

        Defensive: a bad hand-edit (or relay write) must never crash the daemon
        — skip the offending row and keep serving the rest."""
        try:
            vid = str(entry.get("id", "")).strip()
            speaker = str(entry.get("base_speaker") or entry.get("speaker") or "").strip()
            if not vid:
                return None
            if speaker not in known_speakers:
                print(f"custom voice {vid!r}: unknown base_speaker {speaker!r}; skipping")
                return None
            instruct = entry.get("instruct")
            return {
                "id": vid,
                "name": str(entry.get("name") or vid),
                "lang": str(entry.get("lang") or "English"),
                "gender": str(entry.get("gender") or "F"),
                "speaker": speaker,
                "instruct": str(instruct).strip() if instruct else None,
                "gain": _safe_float(entry.get("gain"), 1.0),
                "speed": _safe_float(entry.get("speed"), 1.0),
                "kind": "custom",
            }
        except Exception as e:
            print(f"custom voice entry skipped ({e})")
            return None

    def clone_audio_path(self, ref_audio: str) -> str | None:
        """Resolve a voices.json `ref_audio` value to a real path inside
        `clone_audio_dir`, or None if it lands outside — including via a
        symlink, which is why this resolves before comparing rather than
        after."""
        if not ref_audio:
            return None
        candidate = ref_audio if os.path.isabs(ref_audio) else os.path.join(
            self.clone_audio_dir, ref_audio)
        resolved = os.path.realpath(candidate)
        if resolved == self.clone_audio_dir or resolved.startswith(self.clone_audio_dir + os.sep):
            return resolved
        return None

    def _normalize_clone(self, entry: dict) -> dict | None:
        """Coerce one `clones` row into a clone render spec, or None if unusable.

        A clone voice = a reference recording (`ref_audio`, a 24 kHz WAV path) +
        its transcript (`ref_text`); the Base model reproduces that speaker. We
        require the fields to be present but DON'T stat the file here (it may be
        added after the entry) — synth validates the path at use time. `ref_audio`
        must resolve inside `clone_audio_dir`: voices.json is relay-editable, so
        without that check it could name any file on the box."""
        try:
            vid = str(entry.get("id", "")).strip()
            ref_audio_raw = str(entry.get("ref_audio") or "").strip()
            ref_text = str(entry.get("ref_text") or "").strip()
            if not vid:
                return None
            if not ref_audio_raw or not ref_text:
                print(f"clone voice {vid!r}: needs ref_audio and ref_text; skipping")
                return None
            ref_audio = self.clone_audio_path(ref_audio_raw)
            if ref_audio is None:
                print(f"clone voice {vid!r}: ref_audio must be inside "
                      f"{self.clone_audio_dir}; skipping")
                return None
            return {
                "id": vid,
                "name": str(entry.get("name") or vid),
                "lang": str(entry.get("lang") or "English"),
                "gender": str(entry.get("gender") or "F"),
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "gain": _safe_float(entry.get("gain"), 1.0),
                "speed": _safe_float(entry.get("speed"), 1.0),
                "kind": "clone",
            }
        except Exception as e:
            print(f"clone voice entry skipped ({e})")
            return None

    def _read_dynamic(self) -> tuple:
        """Read voices.json once and return (custom_voices, clone_voices)."""
        try:
            with open(self.custom_voices_path) as f:
                data = json.load(f) or {}
        except FileNotFoundError:
            return [], []
        except (OSError, json.JSONDecodeError) as e:
            print(f"custom voices: cannot read {self.custom_voices_path}: {e}; ignoring")
            return [], []
        known = {v["speaker"] for v in self._builtin}
        custom = [self._normalize_custom(e, known)
                  for e in (data.get("voices") or []) if isinstance(e, dict)]
        clones = [self._normalize_clone(e)
                  for e in (data.get("clones") or []) if isinstance(e, dict)]
        return [v for v in custom if v], [v for v in clones if v]

    def reload_custom(self) -> int:
        """Rebuild the merged registry from built-ins + voices.json (custom +
        clones). Returns the count of dynamic (non-built-in) voices loaded.
        Dynamic ids override built-ins on collision."""
        custom, clones = self._read_dynamic()
        merged, order = {}, []
        for v in self._builtin + custom + clones:
            if v["id"] not in merged:
                order.append(v["id"])
            merged[v["id"]] = v  # dynamic (later) wins on id collision
        voices = [merged[i] for i in order]
        # by_speaker only maps built-in speakers (a raw speaker name resolves to
        # its canonical preset voice, not to a custom/clone voice).
        self._reg = {
            "voices": voices,
            "by_id": {v["id"]: v for v in voices},
            "by_speaker": {v["speaker"]: v for v in voices if v["kind"] == "preset"},
        }
        return len(custom) + len(clones)

    def voice_counts(self) -> dict:
        counts = Counter(v["kind"] for v in self._reg["voices"])
        return {
            "builtin": counts["preset"],
            "custom": counts["custom"],
            "clone": counts["clone"],
            "total": len(self._reg["voices"]),
        }

    def public_voices(self) -> list:
        """The kokoro-compatible {id,name,lang,gender} list for list_voices."""
        return [
            {"id": v["id"], "name": v["name"], "lang": v["lang"], "gender": v["gender"]}
            for v in self._reg["voices"]
        ]

    def resolve(self, voice: str | None) -> dict:
        """Map an incoming `voice` to a concrete render spec.

        Accepts a relayTTS voice id (built-in or custom), a legacy Kokoro id (via
        aliases), or a raw Qwen3 speaker name. Anything unknown falls back to the
        default voice, so old clients and stale preferences never error — the
        drop-in contract. A custom voice carries its own gain/speed defaults so
        it sounds distinct even when the caller sends no overrides."""
        reg = self._reg  # snapshot once: reload may swap it under us
        if not voice:
            voice = self.default_voice
        voice = self.aliases.get(voice, voice)  # legacy Kokoro id -> relayTTS id
        v = (reg["by_id"].get(voice)
             or reg["by_speaker"].get(voice)
             or reg["by_id"][self.default_voice])
        kind = v.get("kind", "preset")
        common = {"kind": kind, "gain": v.get("gain", 1.0), "speed": v.get("speed", 1.0)}
        if kind == "clone":
            # Cloning is driven by the reference recording, not a speaker/instruct.
            return {**common, "ref_audio": v.get("ref_audio"), "ref_text": v.get("ref_text")}
        return {
            **common,
            "speaker": v["speaker"],
            "instruct": v.get("instruct") or self.default_instruct,
        }


# ── Audio post-processing ─────────────────────────────────────────

def _ffmpeg_f32(audio: np.ndarray, sr: int, out_args: list) -> np.ndarray:
    """Run ffmpeg over raw float32 mono PCM via pipes and return the same.

    Piping keeps this off disk entirely and lossless end to end: no container
    to write or parse, and no intermediate quantization below float32.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           *out_args, "-f", "f32le", "-ac", "1", "pipe:1"]
    try:
        result = subprocess.run(cmd, input=audio.astype(np.float32).tobytes(),
                                capture_output=True, check=True, timeout=120)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        detail = (getattr(e, "stderr", None) or b"")[:200]
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        kind = "timed out" if isinstance(e, subprocess.TimeoutExpired) else "failed"
        raise RuntimeError(f"ffmpeg {kind}: {detail}") from None
    return np.frombuffer(result.stdout, dtype=np.float32)


def time_stretch(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Change tempo by `speed` (1.1 = 10% faster) WITHOUT shifting pitch.

    Qwen3's native speed= is a no-op, so we honor the client's `speed` here with
    ffmpeg's `atempo` phase-vocoder. Returns audio unchanged if speed is ~1.0 or
    ffmpeg is missing. atempo handles 0.5-2.0 per pass; chain for anything else.
    """
    # synthesize() already clamps speed to SPEED_RANGE, but this is a public
    # function in its own right: a negative or non-finite speed would otherwise
    # make the halving/doubling loop below spin forever.
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"speed must be a finite positive number, got {speed!r}")
    if abs(speed - 1.0) < 1e-3 or audio.size == 0:
        return audio
    if shutil.which("ffmpeg") is None:
        return audio

    factors, remaining = [], speed
    while remaining > 2.0:
        factors.append(2.0); remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5); remaining /= 0.5
    factors.append(remaining)
    chain = ",".join(f"atempo={f:.5f}" for f in factors)

    return _ffmpeg_f32(audio, sr, ["-filter:a", chain])


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample to `target_sr`. Only used on the remote path, where the server
    is free to answer at a rate other than the one config.yaml declares — every
    client of this daemon is told `sample_rate`, so the audio has to match it."""
    if sr == target_sr or audio.size == 0:
        return audio
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"remote engine returned {sr} Hz but config declares {target_sr} Hz, "
            "and ffmpeg is not installed to resample")

    return _ffmpeg_f32(audio, sr, ["-ar", str(target_sr)])


# ── Remote engine ─────────────────────────────────────────────────

class ClientDisconnected(Exception):
    """The peer closed before sending any part of a request.

    That is what a bare TCP health check looks like — connect, observe the port
    is open, hang up. It is not an error, and logging it as one buries the
    truncated-request case that is.
    """


class RemoteEngine:
    """Synthesis over HTTP against an OpenAI-compatible /v1/audio/speech server.

    The daemon loads no model at all — no MLX, no weights, no generation
    thread. It owns the voice registry, the instruct/speed/gain precedence and
    the time-stretch; only the model call is remote. `base_url` and `model`
    are required — see `pinned_transport.assert_transport_config` for the
    transport rules (TLS, pinning) enforced below.

    If a router fronts the inference server, point `base_url` at the router
    rather than the server: it can hold the upstream credential, so no secret
    has to live beside the daemon. `api_key_env` is there for the case where you
    do talk to a server that authenticates.
    """

    def __init__(self, raw: dict):
        raw = raw or {}
        # URL and model ids come from the environment first: they are
        # deployment facts, not source, so a deployment never has to edit the
        # tracked config to point at its own endpoint.
        env_url = os.environ.get("RELAYTTS_REMOTE_URL")
        self.base_url = (env_url or raw.get("base_url") or "").rstrip("/")
        self.model = os.environ.get("RELAYTTS_REMOTE_MODEL") or raw.get("model") or ""
        # Cloning renders through a different upstream checkpoint (the Base
        # model) than presets do, hence the separate id.
        self.clone_model = (os.environ.get("RELAYTTS_REMOTE_CLONE_MODEL")
                            or raw.get("clone_model") or self.model)
        self.timeout = _safe_float(raw.get("timeout"), 120.0)
        # Read from the environment, never from the config file, so the token
        # is not committed. Unset is normal when a router holds the credential.
        self.api_key = os.environ.get(
            raw.get("api_key_env") or "RELAYTTS_REMOTE_API_KEY") or None

        if not self.model:
            raise ValueError(
                "engine.remote.model is required: set RELAYTTS_REMOTE_MODEL or "
                "engine.remote.model to the id the remote server exposes")

        # Transport security: fail-closed TLS with optional certificate
        # pinning, no skip-verify escape hatch. See pinned_transport.py.
        self.ca_file = os.environ.get("RELAYTTS_REMOTE_CA") or raw.get("ca_file") or None
        self.pins = parse_pins(
            os.environ.get("RELAYTTS_REMOTE_PIN_SHA256") or raw.get("pin_sha256"))
        assert_transport_config(self.base_url, self.ca_file, self.pins)
        self._opener = build_opener(self.base_url, self.ca_file, self.pins)

    @property
    def speech_url(self) -> str:
        return f"{self.base_url}/audio/speech"

    @property
    def label(self) -> str:
        """Endpoint identity for logs and errors — never includes the token."""
        return self.base_url or "<unset>"

    @staticmethod
    def _error_detail(body: bytes) -> str:
        """Pull a human message out of an error body, whatever shape it is."""
        text = (body or b"")[:400].decode("utf-8", "replace").strip()
        try:
            parsed = json.loads(text)
        except ValueError:
            return text
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)
            if err:
                return str(err)
        return text

    def synthesize(self, spec: dict, text: str, lang_code: str,
                   instruct: str | None, temperature: float,
                   sample_rate: int) -> np.ndarray:
        """Render one span remotely and return float32 mono at `sample_rate`.

        `speed` and `gain` are deliberately NOT sent: Qwen3's native speed= is a
        no-op and the server has no gain at all, so both stay local exactly as
        on the local path. A remote daemon therefore renders a given request
        identically to a local one."""
        if spec["kind"] == "clone":
            ref_audio, ref_text = spec.get("ref_audio"), spec.get("ref_text")
            with open(ref_audio, "rb") as f:
                ref_b64 = base64.b64encode(f.read()).decode("ascii")
            payload = {"model": self.clone_model, "ref_audio": ref_b64,
                       "ref_text": ref_text}
        else:
            payload = {"model": self.model, "voice": spec["speaker"]}
            if instruct:
                payload["instructions"] = instruct

        payload.update({"input": text, "response_format": "wav",
                        "language": lang_code, "temperature": temperature})

        req = urllib.request.Request(
            self.speech_url, data=json.dumps(payload).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                wav = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"remote TTS {self.label} returned HTTP {e.code}: "
                f"{self._error_detail(e.read())}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"remote TTS {self.label} unreachable: {e.reason}") from None

        try:
            audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
        except Exception as e:
            raise RuntimeError(
                f"remote TTS {self.label} returned {len(wav)} bytes that are "
                f"not decodable audio: {e}") from None

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return resample(audio, sr, sample_rate)


# ── Daemon ────────────────────────────────────────────────────────

class RelayTTSDaemon:
    def __init__(self, config: Config, host="localhost", port=9997, idle_timeout=0):
        self.cfg = config
        self.host = host
        self.port = port
        self.engine = config.remote
        self.running = False
        self.sock = None
        self.idle_timeout = idle_timeout
        self.last_activity = None
        self.activity_lock = threading.Lock()
        # Relay enhanced-service surface (status + voices.json editor). Set up
        # in start() when relay spawned us; None in standalone mode.
        self._bridge = None
        self._start_time = None
        self._last_rtf = None
        self._voices_mtime = None

    def update_activity(self):
        with self.activity_lock:
            self.last_activity = time.time()

    def check_idle_timeout(self):
        if self.idle_timeout <= 0:
            return False
        with self.activity_lock:
            if self.last_activity is None:
                return False
            return (time.time() - self.last_activity) > self.idle_timeout

    def idle_monitor(self):
        if self.idle_timeout <= 0:
            return
        while self.running:
            if self.check_idle_timeout():
                print(f"Daemon idle for {self.idle_timeout // 60} minutes, shutting down...")
                self.running = False
                break
            time.sleep(30)

    def synthesize(self, text, voice=None, speed=None, lang_code=None, instruct=None, gain=None):
        """Generate speech and return WAV bytes + metadata.

        instruct/gain/speed precedence: explicit request value > the resolved
        voice's own default > global default. Eve omits these when they're at
        their defaults, so an omitted field lets a custom voice's configured
        delivery (instruct + gain + speed) come through; an explicit value from
        the Director still wins per-span."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text too long ({len(text)} > {MAX_TEXT_CHARS} chars)")

        spec = self.cfg.resolve(voice)
        speed = spec["speed"] if speed is None else speed
        gain = spec["gain"] if gain is None else gain
        # Clamp here (not just at the wire edge) so a bad voices.json default
        # is bounded too, not only an explicit per-request value.
        speed = clamp_factor(speed, *SPEED_RANGE, "speed")
        gain = clamp_factor(gain, *GAIN_RANGE, "gain")
        lang_code = lang_code or self.cfg.lang_code

        t0 = time.time()

        if spec["kind"] == "clone":
            # Cloning uses the Base model with reference audio + transcript; the
            # speaker comes from the recording, so voice/instruct don't apply.
            ref_audio, ref_text = spec.get("ref_audio"), spec.get("ref_text")
            if not ref_audio or not os.path.isfile(ref_audio):
                raise RuntimeError(
                    f"clone voice {voice!r}: ref_audio not found: {ref_audio!r}")
            if not ref_text:
                raise RuntimeError(f"clone voice {voice!r}: ref_text is required")
            # The reference recording ships off-box on every request; isfile
            # alone would let an operator point ref_audio at an arbitrary
            # non-audio file and have it read and POSTed.
            try:
                sf.info(ref_audio)
            except Exception:
                raise RuntimeError(
                    f"clone voice {voice!r}: ref_audio is not decodable audio") from None
            descr = f"clone:{os.path.basename(ref_audio)}"
        else:
            instruct = instruct or spec["instruct"]
            descr = f"speaker={spec['speaker']}"

        audio = self.engine.synthesize(
            spec, text, lang_code, instruct, self.cfg.temperature, self.cfg.sample_rate)

        # Qwen3's native speed= is a no-op; honor `speed` with a pitch-preserving
        # time-stretch instead.
        if speed and abs(speed - 1.0) >= 1e-3:
            audio = time_stretch(audio, self.cfg.sample_rate, speed)

        # Amplitude gain makes delivery audible (loud/whisper): instruct= alone
        # barely changes loudness, so the Director sends a gain too. Clip to
        # avoid overflow when boosting.
        if gain and abs(gain - 1.0) >= 1e-3:
            audio = np.clip(audio * gain, -1.0, 1.0)

        generation_time = time.time() - t0
        duration = len(audio) / self.cfg.sample_rate

        buf = io.BytesIO()
        sf.write(buf, audio, self.cfg.sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        rtf = generation_time / duration if duration > 0 else 0
        self._last_rtf = round(rtf, 3)
        print(f"Generated: {duration:.2f}s audio in {generation_time:.2f}s "
              f"(RTF: {rtf:.2f}x) voice={voice or self.cfg.default_voice} {descr}")

        return wav_bytes, {
            "sample_rate": self.cfg.sample_rate,
            "duration": round(duration, 3),
            "generation_time": round(generation_time, 3),
            "rtf": round(rtf, 3),
        }

    # ── TCP protocol (byte-compatible with the Kokoro daemon) ─────

    def _recv_all(self, sock, length, *, allow_empty=False):
        """Read exactly `length` bytes. With allow_empty, a close before the
        first byte raises ClientDisconnected rather than ConnectionError — the
        caller passes it only for the header read, so a close *after* some bytes
        arrived is still the truncation error it has always been."""
        chunks = []
        received = 0
        while received < length:
            chunk = sock.recv(min(length - received, 65536))
            if not chunk:
                if allow_empty and received == 0:
                    raise ClientDisconnected()
                raise ConnectionError("Connection closed")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_request(self, sock):
        """Receive a length-prefixed JSON request (4-byte big-endian header)."""
        header = self._recv_all(sock, 4, allow_empty=True)
        payload_len = struct.unpack("!I", header)[0]

        # A raw-JSON client (first byte is '{' or '[') sent no length prefix at
        # all. Eve only ever uses the length-prefixed framing, so tell a
        # legacy caller plainly rather than groping for a JSON boundary in an
        # unbounded stream.
        if header[0] in (0x7B, 0x5B):
            raise ValueError(
                "raw JSON framing is not supported; send a 4-byte big-endian "
                "length prefix")

        if payload_len > MAX_FRAME_BYTES:
            raise ValueError(f"Payload too large: {payload_len}")
        payload = self._recv_all(sock, payload_len)
        return json.loads(payload.decode("utf-8"))

    def _send_response(self, sock, response_dict):
        """Send a length-prefixed JSON response."""
        data = json.dumps(response_dict).encode("utf-8")
        sock.sendall(struct.pack("!I", len(data)) + data)

    # ── Client handling ───────────────────────────────────────────

    def handle_client(self, client_socket, addr):
        try:
            self.update_activity()
            request = self._recv_request(client_socket)

            # Batch (streaming or not)
            if "batch" in request:
                batch = request["batch"]
                if not isinstance(batch, list):
                    self._send_response(client_socket,
                                        {"success": False, "error": "batch must be a list"})
                    return
                if len(batch) > MAX_BATCH_ITEMS:
                    self._send_response(client_socket, {
                        "success": False,
                        "error": f"batch too large ({len(batch)} > {MAX_BATCH_ITEMS} items)",
                    })
                    return
                if request.get("stream", False):
                    self._handle_batch_streaming(batch, client_socket)
                else:
                    self._handle_batch(batch, client_socket)
                return

            # List voices
            if request.get("action") == "list_voices":
                self._send_response(client_socket,
                                    {"success": True, "voices": self.cfg.public_voices()})
                return

            # Single request. speed/instruct/gain default to None (not 1.0) so an
            # omitted field uses the resolved voice's own default delivery.
            text = request.get("text", "")
            voice = request.get("voice")
            speed = request.get("speed")
            lang_code = request.get("lang_code")
            instruct = request.get("instruct")
            gain = request.get("gain")

            if not text:
                self._send_response(client_socket, {"success": False, "error": "No text provided"})
                return

            wav_bytes, timing = self.synthesize(text, voice, speed, lang_code, instruct, gain)
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

            self._send_response(client_socket, {
                "success": True,
                "audio_base64": audio_b64,
                **timing,
            })

        except ClientDisconnected:
            # A port probe, or a client that hung up before asking anything.
            # Nothing to answer and nothing worth saying.
            return
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            try:
                self._send_response(client_socket, {"success": False, "error": str(e)})
            except Exception:
                pass
        finally:
            client_socket.close()

    def _handle_batch_streaming(self, batch_items, client_socket):
        """Process batch items, streaming each result as newline-delimited JSON."""
        total = len(batch_items)
        successful = 0

        for i, item in enumerate(batch_items):
            try:
                wav_bytes, timing = self.synthesize(
                    text=item.get("text", ""),
                    voice=item.get("voice"),
                    speed=item.get("speed"),
                    lang_code=item.get("lang_code"),
                    instruct=item.get("instruct"),
                    gain=item.get("gain"),
                )
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
                chunk = {
                    "type": "chunk",
                    "index": i,
                    "success": True,
                    "audio_base64": audio_b64,
                    **timing,
                }
                successful += 1
            except Exception as e:
                chunk = {"type": "chunk", "index": i, "success": False, "error": str(e)}

            client_socket.sendall((json.dumps(chunk) + "\n").encode("utf-8"))

        client_socket.sendall(
            (json.dumps({"type": "complete", "total_items": total, "successful_items": successful}) + "\n")
            .encode("utf-8")
        )

    def _handle_batch(self, batch_items, client_socket):
        """Process batch items, return all results at once."""
        results = []
        for i, item in enumerate(batch_items):
            try:
                wav_bytes, timing = self.synthesize(
                    text=item.get("text", ""),
                    voice=item.get("voice"),
                    speed=item.get("speed"),
                    lang_code=item.get("lang_code"),
                    instruct=item.get("instruct"),
                    gain=item.get("gain"),
                )
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
                results.append({"index": i, "success": True, "audio_base64": audio_b64, **timing})
            except Exception as e:
                results.append({"index": i, "success": False, "error": str(e)})

        self._send_response(client_socket, {
            "success": True,
            "batch_results": results,
            "total_items": len(results),
            "successful_items": sum(1 for r in results if r["success"]),
        })

    # ── Server lifecycle ──────────────────────────────────────────

    def start(self):
        self.running = True
        self._start_time = time.time()
        # Nothing to load: no model, no weights, no generation thread. A bad
        # endpoint surfaces per-request rather than blocking startup, so the
        # daemon still serves list_voices if the remote host is booting behind us.
        print(f"Remote engine: {self.engine.label} (model={self.engine.model})")

        # Enhanced-service surface for relay's settings UI (status + voices.json
        # editor). Non-fatal: TTS on 9997 must work even if the inspector doesn't.
        self._start_bridge()
        # Watch voices.json so relay's live edits take effect without a restart.
        try:
            self._voices_mtime = os.path.getmtime(self.cfg.custom_voices_path)
        except OSError:
            self._voices_mtime = None
        threading.Thread(target=self._voices_watch, daemon=True).start()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(1.0)
        self.sock.bind((self.host, self.port))
        self.sock.listen(LISTEN_BACKLOG)

        print(f"relayTTS Daemon started on {self.host}:{self.port}")
        if self.idle_timeout > 0:
            print(f"Auto-shutdown after {self.idle_timeout // 60} minutes idle")
        else:
            print("Idle timeout disabled")

        self.update_activity()

        idle_thread = threading.Thread(target=self.idle_monitor, daemon=True)
        idle_thread.start()

        try:
            while self.running:
                try:
                    client_sock, addr = self.sock.accept()
                    # CPython forces an accepted socket back to blocking mode
                    # when the listener has a timeout, so without setting this
                    # explicitly a stalled peer would hold a handler thread
                    # forever instead of erroring out after CLIENT_TIMEOUT_S.
                    client_sock.settimeout(CLIENT_TIMEOUT_S)
                    t = threading.Thread(target=self.handle_client, args=(client_sock, addr), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
                except socket.error:
                    if self.running:
                        print("Socket error")
                    break
        except KeyboardInterrupt:
            print("\nStopping daemon...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception:
                pass
        if self.sock:
            self.sock.close()
        print("Daemon stopped")

    # ── Relay enhanced-service surface ────────────────────────────

    def _start_bridge(self):
        """Register the manifest with relay and serve /api/status, when relay
        spawned us. Any failure here is logged and swallowed — the daemon still
        serves Eve on 9997."""
        try:
            from relay_bridge import RelayBridge
        except Exception as e:
            print(f"relay bridge module unavailable ({e}); inspector disabled")
            return
        bridge = RelayBridge(
            status_provider=self._status_payload,
            config_path=self.cfg.custom_voices_path,
            speakers=self.cfg.speakers,
            clone_audio_dir=self.cfg.clone_audio_dir,
        )
        if not bridge.enabled:
            print("Standalone mode (no RELAY_BRIDGE_SOCKET); settings inspector disabled")
            return
        try:
            bridge.start()
            self._bridge = bridge
            print("Registered manifest with relay — settings inspector enabled")
        except Exception as e:
            print(f"relay bridge registration failed (non-fatal): {e}")

    def _status_payload(self) -> dict:
        """Read-only snapshot relay polls for the inspector."""
        uptime = round(time.time() - self._start_time, 1) if self._start_time else 0
        return {
            "service": "relaytts-daemon",
            "engine": "remote",
            "endpoint": self.engine.label,
            "model": self.engine.model,
            "port": self.port,
            "sampleRate": self.cfg.sample_rate,
            "defaultVoice": self.cfg.default_voice,
            "voices": self.cfg.voice_counts(),
            "uptimeSeconds": uptime,
            "lastRtf": self._last_rtf,
        }

    def _voices_watch(self):
        """Hot-reload voices.json on change (applyMode=live: relay edits the file
        but does not restart us). Reloading just swaps the registry snapshot."""
        while self.running:
            try:
                mtime = os.path.getmtime(self.cfg.custom_voices_path)
            except OSError:
                mtime = None
            if mtime is not None and mtime != self._voices_mtime:
                self._voices_mtime = mtime
                try:
                    n = self.cfg.reload_custom()
                    print(f"Reloaded custom voices ({n}) from {self.cfg.custom_voices_path}")
                except Exception as e:
                    print(f"custom voices reload failed: {e}")
            time.sleep(1.5)


def main():
    parser = argparse.ArgumentParser(description="relayTTS Daemon (Qwen3-TTS)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9997)
    parser.add_argument("--config", default=os.environ.get("RELAYTTS_CONFIG", DEFAULT_CONFIG_PATH),
                        help="Path to config.yaml (voices + instruct + engine params)")
    parser.add_argument("--custom-voices", default=os.environ.get("RELAYTTS_VOICES"),
                        help="Path to the relay-editable custom voices JSON "
                             "(default: voices.json beside config.yaml)")
    parser.add_argument("--idle-timeout", type=int, default=0,
                        help="Auto-shutdown after idle seconds (0 = disabled)")
    args = parser.parse_args()

    config = Config(args.config, custom_voices_path=args.custom_voices)
    counts = config.voice_counts()
    print(f"Loaded config: {args.config} ({counts['builtin']} built-in + "
          f"{counts['custom']} custom voices, default={config.default_voice})")
    daemon = RelayTTSDaemon(config, host=args.host, port=args.port, idle_timeout=args.idle_timeout)
    daemon.start()


if __name__ == "__main__":
    main()
