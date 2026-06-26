#!/usr/bin/env python3
"""
relayTTS Daemon Server — Qwen3-TTS CustomVoice.

A drop-in replacement for the Kokoro TTS daemon: same length-prefixed JSON TCP
protocol on the same port (9997), so existing clients (Eve) need no changes.
Internally it synthesizes with Qwen3-TTS CustomVoice via mlx-audio (Apple
Silicon / MLX), which gives instruction-driven emotion instead of Kokoro's flat
affect. Returns base64-encoded WAV audio in JSON responses.

Voices and per-voice delivery (`instruct`) live in config.yaml, not in code.
"""
import argparse
import base64
import io
import json
import os
import queue
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import yaml

DAEMON_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(DAEMON_DIR), "config.yaml")


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
        self.repo_id = engine.get(
            "repo_id", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit")
        self.sample_rate = int(engine.get("sample_rate", 24000))
        self.lang_code = engine.get("lang_code", "english")
        self.temperature = float(engine.get("temperature", 0.9))
        # Qwen3 Base checkpoint used for voice cloning (kind="clone"). A separate
        # model from the CustomVoice one above; loaded lazily on first clone use.
        self.clone_repo_id = engine.get(
            "clone_repo_id", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit")

        self.default_voice = raw.get("default_voice", "anna")
        self.default_instruct = raw.get(
            "default_instruct", "Warm, mature and composed.")

        # Built-in voices, in declared order, tagged kind="preset".
        self._builtin = [dict(v, kind="preset") for v in raw.get("voices", [])]
        self.aliases = dict(raw.get("voice_aliases", {}))
        # The model's built-in speakers — the only timbres a custom voice may
        # name. Surfaced to relay's UI as the base_speaker select options.
        self.speakers = [v["speaker"] for v in self._builtin]

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

    def _normalize_clone(self, entry: dict) -> dict | None:
        """Coerce one `clones` row into a clone render spec, or None if unusable.

        A clone voice = a reference recording (`ref_audio`, a 24 kHz WAV path) +
        its transcript (`ref_text`); the Base model reproduces that speaker. We
        require the fields to be present but DON'T stat the file here (it may be
        added after the entry) — synth validates the path at use time."""
        try:
            vid = str(entry.get("id", "")).strip()
            ref_audio = str(entry.get("ref_audio") or "").strip()
            ref_text = entry.get("ref_text")
            ref_text = str(ref_text).strip() if ref_text else ""
            if not vid:
                return None
            if not ref_audio or not ref_text:
                print(f"clone voice {vid!r}: needs ref_audio and ref_text; skipping")
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
        c = {"builtin": 0, "custom": 0, "clone": 0}
        for v in self._reg["voices"]:
            key = {"preset": "builtin", "custom": "custom", "clone": "clone"}.get(
                v["kind"], "builtin")
            c[key] += 1
        c["total"] = len(self._reg["voices"])
        return c

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

def time_stretch(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Change tempo by `speed` (1.1 = 10% faster) WITHOUT shifting pitch.

    Qwen3's native speed= is a no-op, so we honor the client's `speed` here with
    ffmpeg's `atempo` phase-vocoder. Returns audio unchanged if speed is ~1.0 or
    ffmpeg is missing. atempo handles 0.5-2.0 per pass; chain for anything else.
    """
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

    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        sf.write(src, audio, sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-filter:a", chain, dst], check=True)
        out, _ = sf.read(dst, dtype="float32")
    return out.mean(axis=1) if out.ndim > 1 else out


# ── Daemon ────────────────────────────────────────────────────────

class _Job:
    """A unit of model work marshaled from a client thread to the single
    generation thread. `fn` runs on that thread; the caller blocks on `done`."""
    __slots__ = ("fn", "result", "error", "done")

    def __init__(self, fn):
        self.fn = fn
        self.result = None
        self.error = None
        self.done = threading.Event()


class RelayTTSDaemon:
    def __init__(self, config: Config, host="localhost", port=9997, idle_timeout=0):
        self.cfg = config
        self.host = host
        self.port = port
        self.model = None
        self.running = False
        self.sock = None
        self.idle_timeout = idle_timeout
        self.last_activity = None
        self.activity_lock = threading.Lock()
        # mlx-audio (0.4.4) crashes if generate() is invoked from more than one
        # Python thread over the model's lifetime — MLX/Metal binds its state to
        # the first calling thread and a second thread trips
        # "PyThreadState_Get: GIL ... NULL". So ALL model work (load, warmup,
        # every generate) runs on ONE dedicated generation thread; client threads
        # marshal jobs to it via this queue and block for the result. This both
        # satisfies MLX's single-thread requirement and serializes generation.
        self._jobs: "queue.Queue[_Job | None]" = queue.Queue()
        self._ready = threading.Event()
        self._load_ok = False
        self._gen_thread = None
        # Relay enhanced-service surface (status + voices.json editor). Set up
        # in start() when relay spawned us; None in standalone mode.
        self._bridge = None
        self._start_time = None
        self._last_rtf = None
        self._voices_mtime = None
        # Qwen3 Base model for cloning (kind="clone"). Loaded lazily on the
        # generation thread on first clone request, so its RAM is only paid if
        # cloning is actually used. Driven from the same single thread as the
        # CustomVoice model (MLX's cross-thread constraint is per-process, not
        # per-model), so the two coexist safely.
        self._clone_model = None

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

    def load_model(self):
        try:
            print(f"Loading Qwen3-TTS model via mlx-audio ({self.cfg.repo_id})...")
            from mlx_audio.tts.utils import load_model
            self.model = load_model(self.cfg.repo_id)
            print("Qwen3-TTS model loaded!")
            self._warmup_model()
            return True
        except Exception as e:
            print(f"Error loading Qwen3-TTS model: {e}")
            return False

    def _warmup_model(self):
        try:
            print("Warming up model (first-run compilation)...")
            t0 = time.time()
            spec = self.cfg.resolve(self.cfg.default_voice)
            for _ in self.model.generate(
                text="Hello.",
                voice=spec["speaker"],
                instruct=spec["instruct"],
                lang_code=self.cfg.lang_code,
                temperature=self.cfg.temperature,
            ):
                pass  # just trigger compilation
            print(f"Warmup complete ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}")

    def _ensure_clone_model(self):
        """Return the Base model for cloning, loading it on first use. MUST be
        called from the generation thread (it is — invoked inside a _generate
        closure run via _run_on_worker), so MLX loads it on the same thread that
        drives it. The first clone request pays the load + encode cost."""
        if self._clone_model is None:
            print(f"Loading Qwen3-TTS Base (clone) model: {self.cfg.clone_repo_id} ...")
            from mlx_audio.tts.utils import load_model
            self._clone_model = load_model(self.cfg.clone_repo_id)
            print("Clone model loaded.")
        return self._clone_model

    # ── Single generation thread (MLX must be driven from one thread) ──

    def _generation_worker(self):
        """Owns the model for its entire lifetime: loads it, warms it, then
        serves generation jobs off the queue. Running every model call here —
        and only here — is what keeps MLX/Metal from crashing on cross-thread
        access."""
        self._load_ok = self.load_model()
        self._ready.set()
        if not self._load_ok:
            return
        while True:
            job = self._jobs.get()
            if job is None:  # shutdown sentinel
                break
            try:
                job.result = job.fn()
            except Exception as e:  # surfaced to the waiting client thread
                job.error = e
            finally:
                job.done.set()

    def _run_on_worker(self, fn):
        """Submit a callable to the generation thread and block for its result."""
        if not self._load_ok:
            raise RuntimeError("model not loaded")
        job = _Job(fn)
        self._jobs.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def synthesize(self, text, voice=None, speed=None, lang_code=None, instruct=None, gain=None):
        """Generate speech and return WAV bytes + metadata.

        instruct/gain/speed precedence: explicit request value > the resolved
        voice's own default > global default. Eve omits these when they're at
        their defaults, so an omitted field lets a custom voice's configured
        delivery (instruct + gain + speed) come through; an explicit value from
        the Director still wins per-span."""
        spec = self.cfg.resolve(voice)
        speed = spec["speed"] if speed is None else speed
        gain = spec["gain"] if gain is None else gain
        lang_code = lang_code or self.cfg.lang_code

        t0 = time.time()

        # Build the generation closure. It runs on the dedicated generation thread
        # (np.asarray() forces MLX's lazy eval, so the Metal compute happens inside
        # the closure, on that thread). WAV encoding/base64 below operate on local
        # data and stay on the client thread, so overlapping requests can finish
        # encoding in parallel.
        if spec["kind"] == "clone":
            # Cloning uses the Base model with reference audio + transcript; the
            # speaker comes from the recording, so voice/instruct don't apply.
            ref_audio, ref_text = spec.get("ref_audio"), spec.get("ref_text")
            if not ref_audio or not os.path.isfile(ref_audio):
                raise RuntimeError(
                    f"clone voice {voice!r}: ref_audio not found: {ref_audio!r}")
            if not ref_text:
                raise RuntimeError(f"clone voice {voice!r}: ref_text is required")

            def _generate():
                model = self._ensure_clone_model()
                segs = []
                for result in model.generate(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    lang_code=lang_code,
                    temperature=self.cfg.temperature,
                ):
                    segs.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
                return segs

            descr = f"clone:{os.path.basename(ref_audio)}"
        else:
            speaker = spec["speaker"]
            instruct = instruct or spec["instruct"]

            def _generate():
                segs = []
                for result in self.model.generate(
                    text=text,
                    voice=speaker,
                    instruct=instruct,
                    lang_code=lang_code,
                    temperature=self.cfg.temperature,
                ):
                    segs.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
                return segs

            descr = f"speaker={speaker}"

        segments = self._run_on_worker(_generate)

        if not segments:
            raise RuntimeError("No audio generated")

        audio = np.concatenate(segments)
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

    def _recv_all(self, sock, length):
        chunks = []
        received = 0
        while received < length:
            chunk = sock.recv(min(length - received, 65536))
            if not chunk:
                raise ConnectionError("Connection closed")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_request(self, sock):
        """Receive a length-prefixed JSON request (4-byte big-endian header)."""
        header = self._recv_all(sock, 4)
        payload_len = struct.unpack("!I", header)[0]

        # Detect legacy raw-JSON clients (first byte is '{' or '[')
        if header[0] in (0x7B, 0x5B):
            data = header
            while True:
                try:
                    return json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    pass
                chunk = sock.recv(65536)
                if not chunk:
                    return json.loads(data.decode("utf-8"))
                data += chunk

        if payload_len > 100 * 1024 * 1024:
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

            # Batch streaming mode
            if "batch" in request and request.get("stream", False):
                self._handle_batch_streaming(request["batch"], client_socket)
                return

            # Batch non-streaming
            if "batch" in request:
                self._handle_batch(request["batch"], client_socket)
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
        # Load + warm the model on the dedicated generation thread, then wait
        # for it to be ready before we accept connections.
        self.running = True
        self._start_time = time.time()
        self._gen_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self._gen_thread.start()
        self._ready.wait()
        if not self._load_ok:
            self.running = False
            return False

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
        self.sock.listen(5)

        print(f"relayTTS Daemon started on {self.host}:{self.port}")
        if self.idle_timeout > 0:
            print(f"Auto-shutdown after {self.idle_timeout // 60} minutes idle")
        else:
            print("Idle timeout disabled")
        print("Using Apple Silicon MLX acceleration (Qwen3-TTS CustomVoice)")

        self.update_activity()

        idle_thread = threading.Thread(target=self.idle_monitor, daemon=True)
        idle_thread.start()

        try:
            while self.running:
                try:
                    client_sock, addr = self.sock.accept()
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
        self._jobs.put(None)  # wake the generation thread so it can exit
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
        """Read-only snapshot relay polls for the inspector. Counters only —
        must never touch the model (MLX is pinned to the generation thread)."""
        uptime = round(time.time() - self._start_time, 1) if self._start_time else 0
        return {
            "service": "relaytts-daemon",
            "model": self.cfg.repo_id,
            "modelLoaded": self._load_ok,
            "port": self.port,
            "sampleRate": self.cfg.sample_rate,
            "defaultVoice": self.cfg.default_voice,
            "voices": self.cfg.voice_counts(),
            "uptimeSeconds": uptime,
            "lastRtf": self._last_rtf,
        }

    def _voices_watch(self):
        """Hot-reload voices.json on change (applyMode=live: relay edits the file
        but does not restart us). Reloading swaps the registry; no model reload."""
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
