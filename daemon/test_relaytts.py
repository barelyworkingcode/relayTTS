#!/usr/bin/env python3
"""Unit tests for the parts of the daemon that don't need a real upstream:
custom voice loading/merge/resolve precedence, the relay manifest/registration
framing, and the remote engine (transport + pinning) with a faked or
in-process-TLS server. Runs under pytest, or standalone (`python test_relaytts.py`).

Importing relaytts_daemon pulls in numpy/soundfile/yaml (present in the relaytts
env) — there is no model dependency to keep out, since the daemon never loads one.
"""
import base64
import hashlib
import http.server
import io
import json
import os
import shutil
import socket
import socketserver
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.error
from contextlib import contextmanager

import numpy as np

try:
    import pytest
except ImportError:  # the fallback runner at the bottom covers this
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pinned_transport
import relay_bridge
import relaytts_daemon
from relaytts_daemon import Config, RemoteEngine

CONFIG_YAML = """
engine:
  sample_rate: 24000
  lang_code: english
  temperature: 0.9
default_voice: anna
default_instruct: "Default delivery."
voices:
  - {id: anna, name: Anna, lang: English, gender: F, speaker: ono_anna, instruct: "Warm."}
  - {id: ryan, name: Ryan, lang: English, gender: M, speaker: ryan, instruct: "Confident."}
voice_aliases:
  af_heart: anna
  am_michael: ryan
"""


# Config now requires a URL and a model unconditionally, so every _make_config
# call needs a remote block; this is the one tests use unless they override it.
DEFAULT_REMOTE_BLOCK = """
  remote:
    base_url: https://198.51.100.10:8080/v1
    model: test/model
"""

REMOTE_BLOCK = """
  remote:
    base_url: https://198.51.100.10:8080/v1/
    model: upstream/custom-voice
    clone_model: upstream/base
"""


def _make_config(tmp_dir, custom=None, clones=None, remote=None):
    cfg_path = os.path.join(tmp_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        # The remote block belongs under `engine:`, which CONFIG_YAML ends at
        # `temperature`, so appending there keeps the indentation right.
        f.write(CONFIG_YAML.replace(
            "  temperature: 0.9\n",
            "  temperature: 0.9\n" + (remote if remote is not None else DEFAULT_REMOTE_BLOCK)))
    voices_path = os.path.join(tmp_dir, "voices.json")
    if custom is not None or clones is not None:
        with open(voices_path, "w") as f:
            json.dump({"voices": custom or [], "clones": clones or []}, f)
    return Config(cfg_path, custom_voices_path=voices_path)


def test_seeds_empty_voices_file(tmp_path):
    cfg = _make_config(str(tmp_path))
    assert os.path.exists(cfg.custom_voices_path)
    with open(cfg.custom_voices_path) as f:
        assert json.load(f) == {"voices": [], "clones": []}
    assert cfg.voice_counts() == {"builtin": 2, "custom": 0, "clone": 0, "total": 2}


def test_builtin_resolve_defaults(tmp_path):
    cfg = _make_config(str(tmp_path))
    spec = cfg.resolve("ryan")
    assert spec == {"kind": "preset", "speaker": "ryan", "instruct": "Confident.",
                    "gain": 1.0, "speed": 1.0}


def test_alias_and_unknown_fallback(tmp_path):
    cfg = _make_config(str(tmp_path))
    assert cfg.resolve("af_heart")["speaker"] == "ono_anna"   # legacy alias
    assert cfg.resolve("am_michael")["speaker"] == "ryan"
    assert cfg.resolve("ono_anna")["speaker"] == "ono_anna"   # raw speaker name
    assert cfg.resolve("does-not-exist")["speaker"] == "ono_anna"  # -> default
    assert cfg.resolve(None)["speaker"] == "ono_anna"


def test_custom_voice_loaded_and_resolved(tmp_path):
    cfg = _make_config(str(tmp_path), custom=[
        {"id": "narrator", "name": "Narrator", "lang": "English", "gender": "M",
         "base_speaker": "aiden", "instruct": "Gravelly and slow.", "gain": 1.3, "speed": 0.9},
    ])
    # base_speaker aiden is a built-in speaker only if present; config.yaml here
    # only has ono_anna + ryan, so aiden is NOT known -> should be skipped.
    assert cfg.voice_counts()["custom"] == 0


def test_custom_voice_with_known_speaker(tmp_path):
    cfg = _make_config(str(tmp_path), custom=[
        {"id": "narrator", "name": "Narrator", "lang": "English", "gender": "M",
         "base_speaker": "ryan", "instruct": "Gravelly and slow.", "gain": 1.3, "speed": 0.9},
    ])
    assert cfg.voice_counts() == {"builtin": 2, "custom": 1, "clone": 0, "total": 3}
    spec = cfg.resolve("narrator")
    assert spec == {"kind": "custom", "speaker": "ryan", "instruct": "Gravelly and slow.",
                    "gain": 1.3, "speed": 0.9}
    ids = [v["id"] for v in cfg.public_voices()]
    assert ids == ["anna", "ryan", "narrator"]   # built-ins first, in order


def test_custom_overrides_builtin_id(tmp_path):
    cfg = _make_config(str(tmp_path), custom=[
        {"id": "ryan", "name": "Ryan Custom", "base_speaker": "ono_anna",
         "instruct": "Restyled.", "gain": 1.1},
    ])
    # id collision: custom wins, but only one 'ryan' in the list.
    assert cfg.voice_counts()["total"] == 2
    spec = cfg.resolve("ryan")
    assert spec["speaker"] == "ono_anna" and spec["instruct"] == "Restyled."


def test_bad_rows_skipped_not_fatal(tmp_path):
    cfg = _make_config(str(tmp_path), custom=[
        {"name": "No ID"},                                    # missing id
        {"id": "ghost", "base_speaker": "nonexistent"},       # unknown speaker
        "not-a-dict",                                          # wrong type
        {"id": "good", "base_speaker": "ryan", "gain": "loud"},  # bad gain -> 1.0
    ])
    assert cfg.voice_counts()["custom"] == 1
    assert cfg.resolve("good")["gain"] == 1.0   # _safe_float fallback


def test_reload_custom_picks_up_changes(tmp_path):
    cfg = _make_config(str(tmp_path), custom=[])
    assert cfg.voice_counts()["custom"] == 0
    with open(cfg.custom_voices_path, "w") as f:
        json.dump({"voices": [{"id": "x", "base_speaker": "ryan"}]}, f)
    assert cfg.reload_custom() == 1
    assert cfg.voice_counts()["custom"] == 1


def test_speakers_list_for_schema(tmp_path):
    cfg = _make_config(str(tmp_path))
    assert cfg.speakers == ["ono_anna", "ryan"]


# ── clone voices ──────────────────────────────────────────────────

def test_clone_voice_loaded_and_resolved(tmp_path):
    cfg = _make_config(str(tmp_path), clones=[
        {"id": "my_voice", "name": "My Voice", "lang": "English", "gender": "M",
         "ref_audio": "ref.wav", "ref_text": "Hello there.", "gain": 1.1, "speed": 1.0},
    ])
    assert cfg.voice_counts() == {"builtin": 2, "custom": 0, "clone": 1, "total": 3}
    spec = cfg.resolve("my_voice")
    assert spec == {"kind": "clone", "ref_audio": os.path.join(cfg.clone_audio_dir, "ref.wav"),
                    "ref_text": "Hello there.", "gain": 1.1, "speed": 1.0}
    assert [v["id"] for v in cfg.public_voices()] == ["anna", "ryan", "my_voice"]
    # clone speakers must NOT pollute by_speaker (no "speaker" key on clones)
    assert cfg.resolve("ono_anna")["kind"] == "preset"


def test_clone_requires_ref_audio_and_text(tmp_path):
    cfg = _make_config(str(tmp_path), clones=[
        {"id": "no_audio", "ref_text": "hi"},                 # missing ref_audio
        {"id": "no_text", "ref_audio": "x.wav"},              # missing ref_text
        {"ref_audio": "y.wav", "ref_text": "hi"},             # missing id
        {"id": "ok", "ref_audio": "z.wav", "ref_text": "ok"},
    ])
    assert cfg.voice_counts()["clone"] == 1
    assert cfg.resolve("ok")["ref_audio"] == os.path.join(cfg.clone_audio_dir, "z.wav")


def test_custom_and_clone_coexist(tmp_path):
    cfg = _make_config(
        str(tmp_path),
        custom=[{"id": "narrator", "base_speaker": "ryan", "instruct": "Slow."}],
        clones=[{"id": "cloned", "ref_audio": "r.wav", "ref_text": "hi"}],
    )
    assert cfg.voice_counts() == {"builtin": 2, "custom": 1, "clone": 1, "total": 4}


# ── clone audio confinement ─────────────────────────────────────────

def test_clone_audio_path_relative_inside(tmp_path):
    cfg = _make_config(str(tmp_path))
    assert cfg.clone_audio_path("sample.wav") == os.path.join(cfg.clone_audio_dir, "sample.wav")


def test_clone_audio_path_absolute_inside(tmp_path):
    cfg = _make_config(str(tmp_path))
    target = os.path.join(cfg.clone_audio_dir, "sample.wav")
    assert cfg.clone_audio_path(target) == target


def test_clone_audio_path_absolute_outside_rejected(tmp_path):
    cfg = _make_config(str(tmp_path))
    assert cfg.clone_audio_path(str(tmp_path / "elsewhere.wav")) is None


def test_clone_audio_path_symlink_escape_rejected(tmp_path):
    cfg = _make_config(str(tmp_path))
    os.makedirs(cfg.clone_audio_dir, exist_ok=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not audio")
    link = os.path.join(cfg.clone_audio_dir, "escape.wav")
    os.symlink(str(outside), link)
    assert cfg.clone_audio_path("escape.wav") is None


def test_normalize_clone_drops_ref_audio_outside_dir(tmp_path, capsys):
    cfg = _make_config(str(tmp_path), clones=[
        {"id": "escapee", "ref_audio": str(tmp_path / "elsewhere.wav"), "ref_text": "hi"},
    ])
    assert cfg.voice_counts()["clone"] == 0
    assert "must be inside" in capsys.readouterr().out


def test_build_clone_schema_shape():
    schema = relay_bridge.build_clone_schema("/data/clone-audio")
    assert schema[0]["id"] == "clones" and schema[0]["type"] == "array"
    fields = {f["id"]: f for f in schema[0]["item"]["fields"]}
    assert fields["ref_audio"]["required"] is True
    assert "/data/clone-audio" in fields["ref_audio"]["help"]
    assert fields["ref_text"]["type"] == "textarea"
    assert "base_speaker" not in fields and "instruct" not in fields  # not for clones


def test_manifest_includes_both_arrays():
    m = relay_bridge.build_manifest("svc", "/abs/v.json", ["a"], "/data/clone-audio")
    ids = [f["id"] for f in m["config"]["schema"]]
    assert ids == ["voices", "clones"]


# ── relay_bridge pure helpers ─────────────────────────────────────

def test_build_voice_schema_has_speaker_select():
    schema = relay_bridge.build_voice_schema(["a", "b"])
    assert schema[0]["id"] == "voices" and schema[0]["type"] == "array"
    fields = {f["id"]: f for f in schema[0]["item"]["fields"]}
    assert fields["base_speaker"]["type"] == "select"
    assert fields["base_speaker"]["options"] == ["a", "b"]
    assert fields["instruct"]["type"] == "textarea"


def test_build_manifest_shape():
    m = relay_bridge.build_manifest("relaytts-daemon", "/abs/voices.json", ["a"], "/data/clone-audio")
    assert m["routes"] == ["/api/relaytts-daemon/"]   # non-empty (relay requires it)
    assert m["status"]["path"] == "/api/status"
    assert m["config"]["path"] == "/abs/voices.json"
    assert m["config"]["format"] == "json"
    assert m["config"]["applyMode"] == "live"


def test_register_payload_framing():
    m = relay_bridge.build_manifest("svc", "/abs/v.json", ["a"], "/data/clone-audio")
    raw = relay_bridge.build_register_payload("svc", m, "/tmp/i.sock", "itok")
    assert raw.endswith(b"\n")
    msg = json.loads(raw)
    assert msg["type"] == "RegisterManifest"
    assert "token" not in msg
    args = msg["arguments"]
    assert args["serviceId"] == "svc"
    assert args["internalSocket"] == "/tmp/i.sock"
    assert args["internalToken"] == "itok"
    assert args["manifest"]["status"]["path"] == "/api/status"
# ── Remote engine ─────────────────────────────────────────────────

def test_remote_reads_config(tmp_path):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    assert cfg.remote.model == "upstream/custom-voice"
    assert cfg.remote.clone_model == "upstream/base"
    # Trailing slash on base_url must not double up in the joined path.
    assert cfg.remote.speech_url == "https://198.51.100.10:8080/v1/audio/speech"


def test_remote_env_url_overrides_config(tmp_path, monkeypatch):
    """RELAYTTS_REMOTE_URL is a deployment fact, so it overrides whatever
    base_url config.yaml happens to carry, without editing the tracked file."""
    monkeypatch.setenv("RELAYTTS_REMOTE_URL", "https://host:9999/v1")
    cfg = _make_config(str(tmp_path), remote="""
  remote:
    base_url: https://ignored:1/v1
    model: upstream/custom-voice
""")
    assert cfg.remote.speech_url == "https://host:9999/v1/audio/speech"


def test_missing_url_is_fatal(tmp_path):
    """No base_url anywhere (config or env) must fail the daemon at startup,
    not on the first request."""
    try:
        _make_config(str(tmp_path), remote="""
  remote:
    model: upstream/custom-voice
""")
    except ValueError as e:
        assert "base_url" in str(e)
    else:
        raise AssertionError("expected ValueError for a missing base_url")


def test_missing_model_is_fatal(tmp_path):
    """A remote daemon with no model id would otherwise start clean and 400
    on every synthesis; fail at construction instead."""
    try:
        _make_config(str(tmp_path), remote="""
  remote:
    base_url: https://198.51.100.10:8080/v1
""")
    except ValueError as e:
        assert "engine.remote.model" in str(e)
    else:
        raise AssertionError("expected ValueError for remote without a model")


def test_remote_clone_model_falls_back_to_model(tmp_path):
    cfg = _make_config(str(tmp_path), remote="""
  remote:
    base_url: https://198.51.100.10:8080/v1
    model: upstream/only
""")
    assert cfg.remote.clone_model == "upstream/only"


def test_remote_api_key_read_from_named_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TTS_TOKEN", "s3cret")
    cfg = _make_config(str(tmp_path), remote="""
  remote:
    base_url: https://198.51.100.10:8080/v1
    model: upstream/custom-voice
    api_key_env: MY_TTS_TOKEN
""")
    assert cfg.remote.api_key == "s3cret"
    # The token is a secret; the label used in logs and errors must not carry it.
    assert "s3cret" not in cfg.remote.label


def test_error_detail_unwraps_openai_shape():
    detail = RemoteEngine._error_detail(
        b'{"error":{"message":"Model \'x\' not found","type":"invalid_request_error"}}')
    assert detail == "Model 'x' not found"


def test_error_detail_passes_through_non_json():
    assert RemoteEngine._error_detail(b"upstream exploded") == "upstream exploded"


def test_remote_engine_rejects_non_loopback_http():
    try:
        RemoteEngine({"base_url": "http://198.51.100.10:8080/v1", "model": "m"})
    except ValueError as e:
        assert "non-loopback" in str(e)
    else:
        raise AssertionError("expected ValueError for plain http to a non-loopback host")


def test_remote_engine_allows_loopback_http():
    engine = RemoteEngine({"base_url": "http://127.0.0.1:8080/v1", "model": "m"})
    assert engine.speech_url == "http://127.0.0.1:8080/v1/audio/speech"


def _wav_bytes(seconds=0.5, sr=24000):
    import io as _io

    import numpy as np
    import soundfile as sf
    buf = _io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype="float32"), sr,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    """Stands in for the urllib opener RemoteEngine builds, so payload-shape
    tests don't need a real socket. `open_fn(req, timeout=None)` gets the
    exact Request the engine built."""

    def __init__(self, open_fn):
        self._open_fn = open_fn

    def open(self, req, timeout=None):
        return self._open_fn(req, timeout=timeout)


def _capture_request(engine, monkeypatch, body=None):
    """Replace engine._opener and hand back the Request the engine built."""
    seen = {}

    def fake_open(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["payload"] = json.loads(req.data.decode())
        seen["timeout"] = timeout
        return _FakeResponse(body if body is not None else _wav_bytes())

    monkeypatch.setattr(engine, "_opener", _FakeOpener(fake_open))
    return seen


def test_remote_preset_payload_shape(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    seen = _capture_request(cfg.remote, monkeypatch)
    spec = cfg.resolve("ryan")

    audio = cfg.remote.synthesize(spec, "Hello there.", "english",
                                  "Confident.", 0.9, 24000)

    assert seen["url"] == "https://198.51.100.10:8080/v1/audio/speech"
    assert seen["payload"] == {
        "model": "upstream/custom-voice",
        "voice": "ryan",
        "instructions": "Confident.",
        "input": "Hello there.",
        "response_format": "wav",
        "language": "english",
        "temperature": 0.9,
    }
    # speed and gain stay local — sending them would double-apply, since
    # synthesize() still time-stretches and clips on the way out.
    assert "speed" not in seen["payload"] and "gain" not in seen["payload"]
    assert len(audio) == 12000


def test_remote_omits_instructions_when_none(tmp_path, monkeypatch):
    """A null instruct means 'use the server voice's own delivery' — sending
    an empty string instead would flatten the voice's character."""
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    seen = _capture_request(cfg.remote, monkeypatch)
    cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert "instructions" not in seen["payload"]


def test_remote_clone_sends_base64_reference(tmp_path, monkeypatch):
    """The reference recording lives beside the daemon, so it has to travel
    with the request — the remote server has no access to that path."""
    clone_dir = tmp_path / "clone-audio"
    clone_dir.mkdir()
    ref = clone_dir / "ref.wav"
    ref.write_bytes(_wav_bytes(0.25))
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK, clones=[
        {"id": "mine", "name": "Mine", "ref_audio": "ref.wav", "ref_text": "A sample."}])
    seen = _capture_request(cfg.remote, monkeypatch)

    cfg.remote.synthesize(cfg.resolve("mine"), "Hello.", "english", None, 0.9, 24000)

    assert seen["payload"]["model"] == "upstream/base"
    assert seen["payload"]["ref_text"] == "A sample."
    assert base64.b64decode(seen["payload"]["ref_audio"]) == ref.read_bytes()
    assert "voice" not in seen["payload"]


def test_remote_clone_refuses_non_audio_file(tmp_path, monkeypatch):
    """The reference bytes ship upstream on every request, so a non-audio file
    behind ref_audio must be rejected before the opener is ever touched."""
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    os.makedirs(cfg.clone_audio_dir, exist_ok=True)
    with open(os.path.join(cfg.clone_audio_dir, "notaudio.txt"), "w") as f:
        f.write("not audio")
    with open(cfg.custom_voices_path, "w") as f:
        json.dump({"voices": [], "clones": [
            {"id": "bad", "ref_audio": "notaudio.txt", "ref_text": "hi"}]}, f)
    cfg.reload_custom()

    def boom(req, timeout=None):
        raise AssertionError("the opener should not be reached")

    monkeypatch.setattr(cfg.remote, "_opener", _FakeOpener(boom))
    daemon = relaytts_daemon.RelayTTSDaemon(cfg)
    try:
        daemon.synthesize("Hello.", voice="bad")
    except RuntimeError as e:
        assert "not decodable audio" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_remote_sends_bearer_only_when_configured(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    seen = _capture_request(cfg.remote, monkeypatch)
    cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert not any(k.lower() == "authorization" for k in seen["headers"])

    monkeypatch.setenv("RELAYTTS_REMOTE_API_KEY", "tok")
    cfg2 = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    seen2 = _capture_request(cfg2.remote, monkeypatch)
    cfg2.remote.synthesize(cfg2.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert seen2["headers"]["Authorization"] == "Bearer tok"


def test_remote_http_error_names_endpoint_and_reason(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error":{"message":"unknown model"}}'))

    monkeypatch.setattr(cfg.remote, "_opener", _FakeOpener(boom))
    with pytest.raises(RuntimeError) as e:
        cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert "198.51.100.10:8080" in str(e.value)
    assert "HTTP 400" in str(e.value) and "unknown model" in str(e.value)


def test_remote_unreachable_error_is_actionable(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)

    def boom(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(cfg.remote, "_opener", _FakeOpener(boom))
    with pytest.raises(RuntimeError, match="unreachable"):
        cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)


def test_remote_undecodable_body_is_reported(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    _capture_request(cfg.remote, monkeypatch, body=b"<html>proxy error</html>")
    with pytest.raises(RuntimeError, match="not decodable audio"):
        cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)


def test_remote_model_from_env(tmp_path, monkeypatch):
    """The model id is a deployment fact like the URL: env overrides whatever
    config.yaml's remote block happens to declare."""
    monkeypatch.setenv("RELAYTTS_REMOTE_URL", "https://198.51.100.10:8080/v1")
    monkeypatch.setenv("RELAYTTS_REMOTE_MODEL", "router/customvoice")
    cfg = _make_config(str(tmp_path))
    assert cfg.remote.model == "router/customvoice"
    # Clone falls back to the same id rather than erroring, so a deployment
    # that never clones does not have to configure a second model.
    assert cfg.remote.clone_model == "router/customvoice"


def test_remote_clone_model_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAYTTS_REMOTE_URL", "https://198.51.100.10:8080/v1")
    monkeypatch.setenv("RELAYTTS_REMOTE_MODEL", "router/customvoice")
    monkeypatch.setenv("RELAYTTS_REMOTE_CLONE_MODEL", "router/base")
    cfg = _make_config(str(tmp_path))
    assert cfg.remote.clone_model == "router/base"


def test_remote_env_model_overrides_config(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAYTTS_REMOTE_MODEL", "env/wins")
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    assert cfg.remote.model == "env/wins"


def test_remote_ca_and_pins_reach_engine_from_config(tmp_path):
    """A real CA file is needed here (not just a path string): RemoteEngine
    construction runs assert_transport_config, which requires ca_file to
    actually exist and parse as a CA bundle."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    ca = _make_test_ca(tmp_path)
    pin = "AA:BB:" + "cc" * 30  # 64 hex chars once colons are stripped
    cfg = _make_config(str(tmp_path), remote=f"""
  remote:
    base_url: https://198.51.100.10:8080/v1
    model: upstream/custom-voice
    ca_file: "{ca['ca_cert']}"
    pin_sha256: "{pin}"
""")
    assert cfg.remote.ca_file == ca["ca_cert"]
    assert cfg.remote.pins == ["aabb" + "cc" * 30]


def test_remote_env_ca_and_pin_override_config(tmp_path, monkeypatch):
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    ca = _make_test_ca(tmp_path)
    monkeypatch.setenv("RELAYTTS_REMOTE_CA", ca["ca_cert"])
    monkeypatch.setenv("RELAYTTS_REMOTE_PIN_SHA256", "d" * 64)
    cfg = _make_config(str(tmp_path), remote="""
  remote:
    base_url: https://198.51.100.10:8080/v1
    model: upstream/custom-voice
    ca_file: /path/to/ignored-ca.pem
    pin_sha256: "aa"
""")
    assert cfg.remote.ca_file == ca["ca_cert"]
    assert cfg.remote.pins == ["d" * 64]


# ── pinned_transport: assert_transport_config / parse_pins ─────────

def test_assert_transport_config_missing_url_is_fatal():
    try:
        pinned_transport.assert_transport_config("", None, [])
    except ValueError as e:
        assert "base_url" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_assert_transport_config_bad_scheme_is_fatal():
    try:
        pinned_transport.assert_transport_config("ftp://host/v1", None, [])
    except ValueError as e:
        assert "http or https" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_assert_transport_config_http_non_loopback_is_fatal():
    try:
        pinned_transport.assert_transport_config("http://example.com/v1", None, [])
    except ValueError as e:
        assert "non-loopback" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_assert_transport_config_http_loopback_is_ok():
    for netloc in ("127.0.0.1:9999", "localhost:9999", "[::1]:9999"):
        pinned_transport.assert_transport_config(f"http://{netloc}/v1", None, [])


def test_assert_transport_config_http_with_pins_is_fatal():
    try:
        pinned_transport.assert_transport_config(
            "http://127.0.0.1:9999/v1", None, ["a" * 64])
    except ValueError as e:
        assert "http" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_assert_transport_config_http_with_ca_file_is_fatal(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("not a real cert")
    try:
        pinned_transport.assert_transport_config(
            "http://127.0.0.1:9999/v1", str(ca), [])
    except ValueError as e:
        assert "http" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_assert_transport_config_missing_ca_file_is_fatal(tmp_path):
    try:
        pinned_transport.assert_transport_config(
            "https://host/v1", str(tmp_path / "missing.pem"), [])
    except ValueError as e:
        assert "ca_file" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_assert_transport_config_malformed_pin_is_fatal():
    try:
        pinned_transport.assert_transport_config("https://host/v1", None, ["not-a-fingerprint"])
    except ValueError as e:
        assert "pin_sha256" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_assert_transport_config_https_with_pin_is_ok():
    pinned_transport.assert_transport_config("https://host/v1", None, ["a" * 64])


def test_parse_pins_normalizes_string_with_colons():
    pin = "AA:BB:CC:" + "dd" * 28 + "EE"  # 64 hex chars once colons are stripped
    assert pinned_transport.parse_pins(pin) == [pin.replace(":", "").lower()]


def test_parse_pins_accepts_list_input():
    pins = ["A" * 64, "b" * 64]
    assert pinned_transport.parse_pins(pins) == ["a" * 64, "b" * 64]


def test_parse_pins_empty_input_yields_no_pins():
    assert pinned_transport.parse_pins(None) == []
    assert pinned_transport.parse_pins("") == []


# ── pinned_transport: real TLS, in-process ──────────────────────────
#
# A real CA + two leaf certs via the openssl CLI, and a real TLS server in a
# background thread, so the pinning path is exercised against actual
# handshakes and actual fingerprints rather than mocks of the ssl module.

def _run_openssl(*args):
    subprocess.run(["openssl", *args], check=True, capture_output=True)


def _make_test_ca(tmp_path):
    """A self-signed CA plus two leaf certs (distinct keys), both signed by
    that CA and valid for localhost/127.0.0.1. Returns the CA cert path and a
    list of {cert, key, fingerprint} for the two leaves."""
    ca_key = tmp_path / "ca.key"
    ca_cert = tmp_path / "ca.crt"
    _run_openssl("genrsa", "-out", str(ca_key), "2048")
    _run_openssl("req", "-x509", "-new", "-nodes", "-key", str(ca_key),
                 "-sha256", "-days", "2", "-out", str(ca_cert),
                 "-subj", "/CN=relaytts-test-ca")

    # Hostname verification needs a SAN, not just a CN.
    ext_file = tmp_path / "leaf.ext"
    ext_file.write_text("subjectAltName=DNS:localhost,IP:127.0.0.1\n")

    leaves = []
    for i in (1, 2):
        key = tmp_path / f"leaf{i}.key"
        csr = tmp_path / f"leaf{i}.csr"
        cert = tmp_path / f"leaf{i}.crt"
        _run_openssl("genrsa", "-out", str(key), "2048")
        _run_openssl("req", "-new", "-key", str(key), "-out", str(csr),
                     "-subj", f"/CN=leaf{i}.relaytts-test")
        _run_openssl("x509", "-req", "-in", str(csr), "-CA", str(ca_cert),
                     "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(cert),
                     "-days", "2", "-sha256", "-extfile", str(ext_file))
        der = ssl.PEM_cert_to_DER_cert(cert.read_text())
        leaves.append({
            "cert": str(cert), "key": str(key),
            "fingerprint": hashlib.sha256(der).hexdigest(),
        })
    return {"ca_cert": str(ca_cert), "leaves": leaves}


class _SpeechHandler(http.server.BaseHTTPRequestHandler):
    """Answers POST /v1/audio/speech with a short silent WAV, so
    RemoteEngine.synthesize can complete end to end over the real handshake."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)  # drain the request body
        body = _wav_bytes(0.1, 24000)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _start_tls_server(cert_path, key_path):
    """Start the speech handler over TLS on an ephemeral loopback port."""
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _SpeechHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


_PRESET_SPEC = {"kind": "preset", "speaker": "ryan"}


def test_tls_ca_verified_no_pins_succeeds(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    ca = _make_test_ca(tmp_path)
    leaf = ca["leaves"][0]
    httpd, port = _start_tls_server(leaf["cert"], leaf["key"])
    try:
        engine = RemoteEngine({"base_url": f"https://127.0.0.1:{port}/v1",
                               "model": "test/model", "ca_file": ca["ca_cert"]})
        audio = engine.synthesize(_PRESET_SPEC, "hi", "english", None, 0.9, 24000)
        assert len(audio) > 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_tls_without_ca_file_fails_verification(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    ca = _make_test_ca(tmp_path)
    leaf = ca["leaves"][0]
    httpd, port = _start_tls_server(leaf["cert"], leaf["key"])
    try:
        # No ca_file: only system roots are trusted, and this CA isn't one.
        engine = RemoteEngine({"base_url": f"https://127.0.0.1:{port}/v1", "model": "test/model"})
        with pytest.raises(RuntimeError) as e:
            engine.synthesize(_PRESET_SPEC, "hi", "english", None, 0.9, 24000)
        assert f"127.0.0.1:{port}" in str(e.value)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_tls_pin_matches_serving_leaf_succeeds(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    ca = _make_test_ca(tmp_path)
    leaf = ca["leaves"][0]
    httpd, port = _start_tls_server(leaf["cert"], leaf["key"])
    try:
        engine = RemoteEngine({"base_url": f"https://127.0.0.1:{port}/v1",
                               "model": "test/model", "ca_file": ca["ca_cert"],
                               "pin_sha256": leaf["fingerprint"]})
        audio = engine.synthesize(_PRESET_SPEC, "hi", "english", None, 0.9, 24000)
        assert len(audio) > 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_tls_pin_mismatch_rejects_valid_cert_from_wrong_leaf(tmp_path):
    """The MITM-with-a-valid-cert case: leaf2 is signed by the same trusted CA
    as leaf1, so cert validation alone would pass it — the pin is what catches
    the swap."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    ca = _make_test_ca(tmp_path)
    leaf1, leaf2 = ca["leaves"]
    httpd, port = _start_tls_server(leaf2["cert"], leaf2["key"])
    try:
        engine = RemoteEngine({"base_url": f"https://127.0.0.1:{port}/v1",
                               "model": "test/model", "ca_file": ca["ca_cert"],
                               "pin_sha256": leaf1["fingerprint"]})
        with pytest.raises(RuntimeError) as e:
            engine.synthesize(_PRESET_SPEC, "hi", "english", None, 0.9, 24000)
        assert "not pinned" in str(e.value)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ── unix:<path> transport: relay's model.sock ──────────────────────
#
# A real OpenAI-compatible HTTP server on a real AF_UNIX socket — no mock of
# the socket layer — mirroring _FakeBridge's pattern above. `responses` is a
# list of (status, error_body_or_None) consumed in order; once exhausted the
# last entry repeats, so a test that wants "always 503" just passes one entry.

class _FakeModelSockHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so the reuse test means something

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        self.server.record({
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "payload": payload,
        })
        status, error_body = self.server.next_response()
        if status == 200:
            wav = _wav_bytes(0.1, 24000)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
        else:
            body = json.dumps(error_body or {"error": {"message": "denied"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):
        pass


class _FakeModelSockServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, sock_path, responses):
        self._lock = threading.Lock()
        self.requests = []
        self.connection_count = 0
        self.responses = list(responses)
        super().__init__(sock_path, _FakeModelSockHandler)

    def get_request(self):
        conn, addr = super().get_request()
        with self._lock:
            self.connection_count += 1
        return conn, addr

    def record(self, entry):
        with self._lock:
            self.requests.append(entry)

    def next_response(self):
        with self._lock:
            if len(self.responses) > 1:
                return self.responses.pop(0)
            return self.responses[0]


@contextmanager
def _model_sock_server(responses=((200, None),)):
    # AF_UNIX paths cap at ~104 bytes on macOS; pytest's tmp_path is too deep
    # (see _FakeBridge above), so this gets its own short-lived base under /tmp.
    d = tempfile.mkdtemp(prefix="rtts-model-")
    sock_path = os.path.join(d, "model.sock")
    srv = _FakeModelSockServer(sock_path, responses)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv, sock_path
    finally:
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(d, ignore_errors=True)


def test_unix_url_relative_path_refused():
    for bad in ("unix:relative/path.sock", "unix:", "unix:~/model.sock"):
        try:
            pinned_transport.assert_transport_config(bad, None, [])
        except ValueError as e:
            assert "absolute path" in str(e)
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_unix_url_absolute_path_is_ok():
    pinned_transport.assert_transport_config("unix:/tmp/model.sock", None, [])


def test_unix_url_with_ca_file_is_fatal(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("irrelevant — rejected before it would be read")
    try:
        pinned_transport.assert_transport_config("unix:/tmp/model.sock", str(ca), [])
    except ValueError as e:
        assert "unix" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_unix_url_with_pins_is_fatal():
    try:
        pinned_transport.assert_transport_config("unix:/tmp/model.sock", None, ["a" * 64])
    except ValueError as e:
        assert "unix" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_remote_engine_rejects_relative_unix_path():
    try:
        RemoteEngine({"base_url": "unix:relative.sock", "model": "m"})
    except ValueError as e:
        assert "absolute path" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_unix_transport_sends_expected_request_no_auth_header():
    with _model_sock_server() as (srv, sock_path):
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "test/model"})
        assert engine.speech_url == "unix://model.sock/v1/audio/speech"
        audio = engine.synthesize({"kind": "preset", "speaker": "ryan"}, "Hello.",
                                  "english", "Confident.", 0.9, 24000)
        assert len(audio) > 0
        assert len(srv.requests) == 1
        req = srv.requests[0]
        assert req["path"] == "/v1/audio/speech"
        assert req["payload"]["model"] == "test/model"
        assert req["payload"]["voice"] == "ryan"
        assert "authorization" not in req["headers"]
        assert "x-api-key" not in req["headers"]


def test_unix_transport_ignores_configured_api_key_with_warning(monkeypatch, capsys):
    with _model_sock_server() as (srv, sock_path):
        monkeypatch.setenv("RELAYTTS_REMOTE_API_KEY", "s3cret-value")
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "m"})
        out = capsys.readouterr().out
        assert "RELAYTTS_REMOTE_API_KEY" in out
        assert "s3cret-value" not in out
        assert engine.api_key is None

        engine.synthesize({"kind": "preset", "speaker": "ryan"}, "Hi.", "english", None, 0.9, 24000)
        assert "authorization" not in srv.requests[0]["headers"]


def test_unix_transport_api_key_canary_never_sent_or_logged(monkeypatch, capsys):
    """The value must not leak into a request header or anything printed,
    whatever the daemon does with it internally."""
    canary = "CANARY-" + "f" * 40
    with _model_sock_server() as (srv, sock_path):
        monkeypatch.setenv("RELAYTTS_REMOTE_API_KEY", canary)
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "m"})
        engine.synthesize({"kind": "preset", "speaker": "ryan"}, "Hi.", "english", None, 0.9, 24000)
    out = capsys.readouterr().out
    assert canary not in out
    assert canary not in json.dumps(srv.requests)


def test_https_config_still_sends_bearer_when_configured(tmp_path, monkeypatch):
    """Guards against the unix-only api_key suppression leaking onto the
    https path — same assertion as test_remote_sends_bearer_only_when_configured,
    kept here as a contrast to the unix behaviour above."""
    monkeypatch.setenv("RELAYTTS_REMOTE_API_KEY", "tok")
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    seen = _capture_request(cfg.remote, monkeypatch)
    cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert seen["headers"]["Authorization"] == "Bearer tok"


def test_unix_transport_401_gives_clear_message():
    with _model_sock_server(responses=[(401, {"error": "unauthorized"})]) as (srv, sock_path):
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "m"})
        with pytest.raises(RuntimeError, match="not authorised by relay.*models capability"):
            engine.synthesize({"kind": "preset", "speaker": "ryan"}, "hi", "english", None, 0.9, 24000)


def test_unix_transport_404_gives_clear_message():
    with _model_sock_server(responses=[(404, {"error": "not found"})]) as (srv, sock_path):
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "m"})
        with pytest.raises(RuntimeError, match="model not allowed or unknown"):
            engine.synthesize({"kind": "preset", "speaker": "ryan"}, "hi", "english", None, 0.9, 24000)


def test_unix_transport_429_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(relaytts_daemon.time, "sleep", lambda s: None)
    responses = [(429, {"error": "rate_limited"}), (429, {"error": "rate_limited"}), (200, None)]
    with _model_sock_server(responses=responses) as (srv, sock_path):
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "m"})
        audio = engine.synthesize({"kind": "preset", "speaker": "ryan"}, "hi", "english", None, 0.9, 24000)
        assert len(audio) > 0
        assert len(srv.requests) == 3


def test_unix_transport_503_retries_then_gives_up(monkeypatch):
    monkeypatch.setattr(relaytts_daemon.time, "sleep", lambda s: None)
    with _model_sock_server(responses=[(503, {"error": "model host unavailable"})]) as (srv, sock_path):
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "m"})
        with pytest.raises(RuntimeError, match="HTTP 503"):
            engine.synthesize({"kind": "preset", "speaker": "ryan"}, "hi", "english", None, 0.9, 24000)
        assert len(srv.requests) == RemoteEngine._MAX_ATTEMPTS


def test_unix_transport_reuses_connection_across_requests():
    with _model_sock_server() as (srv, sock_path):
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "m"})
        for _ in range(20):
            engine.synthesize({"kind": "preset", "speaker": "ryan"}, "hi", "english", None, 0.9, 24000)
        assert len(srv.requests) == 20
        # Bounded well under N: a fresh connection per call would be 20.
        assert srv.connection_count <= 2


def test_unix_transport_decodes_chunked_response():
    """Transfer-Encoding: chunked, written by hand — exercises the same
    http.client chunked-decoding path a streaming upstream would use."""
    class _ChunkedHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            wav = _wav_bytes(0.1, 24000)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            mid = len(wav) // 2
            for chunk in (wav[:mid], wav[mid:]):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *a):
            pass

    d = tempfile.mkdtemp(prefix="rtts-model-chunked-")
    sock_path = os.path.join(d, "model.sock")
    srv = socketserver.ThreadingUnixStreamServer(sock_path, _ChunkedHandler)
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        engine = RemoteEngine({"base_url": f"unix:{sock_path}", "model": "m"})
        audio = engine.synthesize({"kind": "preset", "speaker": "ryan"}, "hi", "english", None, 0.9, 24000)
        assert len(audio) > 0
    finally:
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(d, ignore_errors=True)


# ── framing: probe vs truncation ──────────────────────────────────

class _FakeSock:
    """Serves a scripted byte stream, then behaves like a closed peer. Also
    records everything sent back, for tests that need to inspect the
    response `handle_client` writes."""

    def __init__(self, data=b""):
        self._data = data
        self.sent = b""

    def recv(self, n):
        if not self._data:
            return b""
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk

    def sendall(self, data):
        self.sent += data

    def close(self):
        pass


def _daemon(tmp_path):
    return relaytts_daemon.RelayTTSDaemon(_make_config(str(tmp_path)))


def _framed(obj) -> bytes:
    data = json.dumps(obj).encode("utf-8")
    return struct.pack("!I", len(data)) + data


def _decode_response(sent: bytes) -> dict:
    length = struct.unpack("!I", sent[:4])[0]
    return json.loads(sent[4:4 + length].decode("utf-8"))


def test_probe_disconnect_is_not_an_error(tmp_path):
    """A health check opens the port and hangs up without sending. Normal —
    and logging it as an error buries the truncated-request case that is not."""
    try:
        _daemon(tmp_path)._recv_request(_FakeSock(b""))
    except relaytts_daemon.ClientDisconnected:
        pass
    else:
        raise AssertionError("expected ClientDisconnected for a bare probe")


def test_truncated_request_is_still_an_error(tmp_path):
    try:
        _daemon(tmp_path)._recv_request(_FakeSock(struct.pack(">I", 100) + b"abc"))
    except relaytts_daemon.ClientDisconnected:
        raise AssertionError("a truncated request must not be treated as a probe")
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError for a truncated request")


def test_header_only_close_is_truncation_not_probe(tmp_path):
    try:
        _daemon(tmp_path)._recv_request(_FakeSock(struct.pack(">I", 50)))
    except relaytts_daemon.ClientDisconnected:
        raise AssertionError("a close after the header is truncation, not a probe")
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError")


def test_frame_over_max_size_rejected(tmp_path):
    huge = relaytts_daemon.MAX_FRAME_BYTES + 1
    try:
        _daemon(tmp_path)._recv_request(_FakeSock(struct.pack(">I", huge)))
    except ValueError as e:
        assert "Payload too large" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_raw_json_framing_is_a_clear_error(tmp_path):
    try:
        _daemon(tmp_path)._recv_request(_FakeSock(b'{"text": "hi"}'))
    except ValueError as e:
        assert "raw JSON framing is not supported" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ── clamp_factor ─────────────────────────────────────────────────

def test_clamp_factor_none_passthrough():
    assert relaytts_daemon.clamp_factor(None, 0.25, 4.0, "speed") is None


def test_clamp_factor_clamps_both_ends():
    assert relaytts_daemon.clamp_factor(-1, 0.25, 4.0, "speed") == 0.25
    assert relaytts_daemon.clamp_factor(100, 0.25, 4.0, "speed") == 4.0
    assert relaytts_daemon.clamp_factor(1.5, 0.25, 4.0, "speed") == 1.5


def test_clamp_factor_rejects_bool():
    try:
        relaytts_daemon.clamp_factor(True, 0.0, 4.0, "gain")
    except ValueError as e:
        assert "gain must be a number" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_clamp_factor_rejects_non_numeric():
    try:
        relaytts_daemon.clamp_factor("fast", 0.25, 4.0, "speed")
    except ValueError as e:
        assert "speed must be a number" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_clamp_factor_rejects_nan_and_infinity():
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            relaytts_daemon.clamp_factor(bad, 0.25, 4.0, "speed")
        except ValueError as e:
            assert "finite" in str(e)
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


# ── synthesize()-level bounds (the crash fix) ───────────────────────

def _remote_daemon(tmp_path, monkeypatch):
    """A daemon whose remote engine is stubbed to return a short clip, so
    synthesize() runs its full text/speed/gain validation without a real
    upstream call."""
    daemon = _daemon(tmp_path)
    monkeypatch.setattr(daemon.engine, "synthesize",
                        lambda *a, **k: np.zeros(2400, dtype=np.float32))
    return daemon


def test_synthesize_clamps_negative_speed_instead_of_hanging(tmp_path, monkeypatch):
    daemon = _remote_daemon(tmp_path, monkeypatch)
    seen = {}
    real_stretch = relaytts_daemon.time_stretch

    def spy_stretch(audio, sr, speed):
        seen["speed"] = speed
        return real_stretch(audio, sr, speed)

    monkeypatch.setattr(relaytts_daemon, "time_stretch", spy_stretch)

    wav_bytes, timing = daemon.synthesize("Hello.", speed=-1)
    assert wav_bytes
    assert seen["speed"] == relaytts_daemon.SPEED_RANGE[0]


def test_synthesize_rejects_infinite_speed_instead_of_hanging(tmp_path, monkeypatch):
    daemon = _remote_daemon(tmp_path, monkeypatch)
    try:
        daemon.synthesize("Hello.", speed=json.loads("Infinity"))
    except ValueError as e:
        assert "finite" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_synthesize_clamps_gain(tmp_path, monkeypatch):
    daemon = _remote_daemon(tmp_path, monkeypatch)
    wav_bytes, timing = daemon.synthesize("Hello.", gain=999)
    assert wav_bytes  # clipped, not rejected or hung


def test_synthesize_rejects_non_string_text(tmp_path):
    try:
        _daemon(tmp_path).synthesize(12345)
    except ValueError as e:
        assert "text must be a string" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_synthesize_rejects_text_too_long(tmp_path):
    try:
        _daemon(tmp_path).synthesize("x" * (relaytts_daemon.MAX_TEXT_CHARS + 1))
    except ValueError as e:
        assert "text too long" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_time_stretch_rejects_bad_speed_on_its_own():
    for bad in (0, -1, float("inf"), float("nan")):
        try:
            relaytts_daemon.time_stretch(np.zeros(10, dtype=np.float32), 24000, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for speed={bad!r}")


# ── batch bounds ─────────────────────────────────────────────────

def test_batch_not_a_list_returns_error(tmp_path):
    daemon = _daemon(tmp_path)
    sock = _FakeSock(_framed({"batch": "not-a-list"}))
    daemon.handle_client(sock, ("127.0.0.1", 0))
    assert _decode_response(sock.sent) == {"success": False, "error": "batch must be a list"}


def test_batch_too_many_items_returns_error(tmp_path):
    daemon = _daemon(tmp_path)
    batch = [{"text": "hi"}] * (relaytts_daemon.MAX_BATCH_ITEMS + 1)
    sock = _FakeSock(_framed({"batch": batch}))
    daemon.handle_client(sock, ("127.0.0.1", 0))
    resp = _decode_response(sock.sent)
    assert resp["success"] is False
    assert "batch too large" in resp["error"]


# ── stale bridge dir sweep ───────────────────────────────────────

def test_sweep_stale_bridge_dirs_removes_dead_and_keeps_live(monkeypatch):
    import tempfile as _tempfile

    # pytest's tmp_path is too deep for an AF_UNIX socket path (~104 byte
    # limit on macOS), so this needs its own short-lived base under /tmp.
    base = _tempfile.mkdtemp(prefix="rtts-sweep-")
    monkeypatch.setattr(relay_bridge.tempfile, "gettempdir", lambda: base)

    dead_dir = os.path.join(base, "relaytts-bridge-dead")
    os.mkdir(dead_dir)  # no internal.sock -> connect fails -> removed

    live_dir = os.path.join(base, "relaytts-bridge-live")
    os.mkdir(live_dir)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(os.path.join(live_dir, "internal.sock"))
    srv.listen(1)
    try:
        relay_bridge._sweep_stale_bridge_dirs()
        assert not os.path.isdir(dead_dir)
        assert os.path.isdir(live_dir)
    finally:
        srv.close()
        shutil.rmtree(base, ignore_errors=True)


# ── ffmpeg via pipes ─────────────────────────────────────────────

def _sine(seconds, sr, hz=440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_time_stretch_changes_length_by_speed():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    sr = 24000
    audio = _sine(0.2, sr)
    speed = 1.5
    out = relaytts_daemon.time_stretch(audio, sr, speed)
    expected = len(audio) / speed
    assert abs(len(out) - expected) / expected < 0.05


def test_resample_changes_length_by_ratio():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    sr, target_sr = 24000, 16000
    audio = _sine(0.2, sr)
    out = relaytts_daemon.resample(audio, sr, target_sr)
    expected = len(audio) * target_sr / sr
    assert abs(len(out) - expected) / expected < 0.05


# ── relay launch identity (fd 3 secret + Hello) ──────────────────

SECRET = "0123456789abcdef" * 4


def _pipe_with(data: bytes) -> int:
    r, w = os.pipe()
    os.write(w, data)
    os.close(w)
    return r


def _fd_is_closed(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return True
    return False


def test_read_launch_secret_valid_and_closes_fd():
    fd = _pipe_with(SECRET.encode())
    assert relay_bridge.read_launch_secret(fd) == SECRET
    assert _fd_is_closed(fd)


def test_read_launch_secret_multiple_writes_read_to_eof():
    r, w = os.pipe()
    os.write(w, SECRET[:10].encode())
    os.write(w, SECRET[10:].encode())
    os.close(w)
    assert relay_bridge.read_launch_secret(r) == SECRET


def test_read_launch_secret_rejects_malformed_without_echoing():
    bad = [SECRET + "\n", SECRET.upper(), SECRET[:63], SECRET + "0", "",
           ("g" * 64), "a" * 5000]
    for value in bad:
        fd = _pipe_with(value.encode())
        try:
            relay_bridge.read_launch_secret(fd)
        except relay_bridge.LaunchIdentityError as e:
            assert SECRET not in str(e) and SECRET.upper() not in str(e)
        else:
            raise AssertionError(f"accepted malformed secret of length {len(value)}")
        assert _fd_is_closed(fd)


def test_read_launch_secret_bad_fd_fails_closed():
    r, w = os.pipe()
    os.close(r)
    os.close(w)
    try:
        relay_bridge.read_launch_secret(r)
    except relay_bridge.LaunchIdentityError:
        pass
    else:
        raise AssertionError("closed fd accepted")


class _FakeBridge:
    """A real Unix-socket server that records each request line and answers
    with `reply(request_dict)` (bytes, or None to close without answering)."""

    def __init__(self, reply):
        import tempfile as _tempfile
        # AF_UNIX paths cap at ~104 bytes on macOS, so not pytest's tmp_path.
        self.dir = _tempfile.mkdtemp(prefix="rtts-br-")
        self.path = os.path.join(self.dir, "bridge.sock")
        self.reply = reply
        self.requests = []
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(4)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                req = json.loads(buf.split(b"\n", 1)[0])
                self.requests.append(req)
                out = self.reply(req)
                if out is not None:
                    conn.sendall(out)

    def close(self):
        self.srv.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def _ok(service_id="relaytts-daemon", relay_pid=4242):
    return lambda req: json.dumps({"type": "OK", "data": {
        "service_id": service_id, "relay_pid": relay_pid}}).encode() + b"\n"


def _hello_raises(reply, secret=SECRET):
    br = _FakeBridge(reply)
    try:
        relay_bridge.send_hello(br.path, "relaytts-daemon", secret, timeout=2.0)
    except relay_bridge.LaunchIdentityError as e:
        assert secret not in str(e)
        return str(e)
    finally:
        br.close()
    raise AssertionError("Hello unexpectedly succeeded")


def test_hello_success_sends_exact_frame():
    br = _FakeBridge(_ok())
    try:
        data = relay_bridge.send_hello(br.path, "relaytts-daemon", SECRET, timeout=2.0)
    finally:
        br.close()
    assert data == {"service_id": "relaytts-daemon", "relay_pid": 4242}
    assert br.requests == [{"type": "Hello", "name": "relaytts-daemon", "token": SECRET}]


def test_hello_error_frame_fails_closed():
    msg = _hello_raises(lambda req: b'{"type":"Error","code":-32001,"message":"unauthorized"}\n')
    assert "-32001" in msg


def test_hello_malformed_responses_fail_closed():
    for reply in (
        lambda req: b"not json\n",
        lambda req: b"[1,2]\n",
        lambda req: b'{"type":"OK"}\n',
        lambda req: b'{"type":"Result","data":{"service_id":"relaytts-daemon","relay_pid":1}}\n',
        lambda req: b'{"type":"OK","data":{"relay_pid":1}}\n',
        lambda req: b'{"type":"OK","data":{"service_id":"relaytts-daemon","relay_pid":"1"}}\n',
        _ok(service_id="someone-else"),
        lambda req: None,
    ):
        _hello_raises(reply)


def test_hello_unreachable_socket_fails_closed():
    try:
        relay_bridge.send_hello("/tmp/rtts-no-such-bridge.sock", "svc", SECRET, timeout=1.0)
    except relay_bridge.LaunchIdentityError as e:
        assert SECRET not in str(e)
    else:
        raise AssertionError("unreachable bridge accepted")


def test_establish_unset_is_standalone():
    assert relay_bridge.establish_launch_identity({"RELAY_BRIDGE_SOCKET": "/x"}) is False


def test_establish_full_handshake_and_env_scrubbed():
    br = _FakeBridge(_ok())
    fd = _pipe_with(SECRET.encode())
    env = {"RELAY_LAUNCH_FD": str(fd), "RELAY_BRIDGE_SOCKET": br.path,
           "RELAY_SERVICE_ID": "relaytts-daemon"}
    try:
        assert relay_bridge.establish_launch_identity(env) is True
    finally:
        br.close()
    assert "RELAY_LAUNCH_FD" not in env
    assert _fd_is_closed(fd)
    assert br.requests[0]["token"] == SECRET


def test_establish_fails_closed_on_bad_inputs():
    for build in (
        lambda: {"RELAY_LAUNCH_FD": "three", "RELAY_BRIDGE_SOCKET": "/x", "RELAY_SERVICE_ID": "s"},
        lambda: {"RELAY_LAUNCH_FD": str(_pipe_with(b"short")),
                 "RELAY_BRIDGE_SOCKET": "/x", "RELAY_SERVICE_ID": "s"},
        lambda: {"RELAY_LAUNCH_FD": str(_pipe_with(SECRET.encode())), "RELAY_SERVICE_ID": "s"},
        lambda: {"RELAY_LAUNCH_FD": str(_pipe_with(SECRET.encode())),
                 "RELAY_BRIDGE_SOCKET": "/tmp/rtts-no-such-bridge.sock", "RELAY_SERVICE_ID": "s"},
    ):
        env = build()
        try:
            relay_bridge.establish_launch_identity(env)
        except relay_bridge.LaunchIdentityError as e:
            assert SECRET not in str(e)
            assert "RELAY_LAUNCH_FD" not in env
        else:
            raise AssertionError(f"accepted {sorted(env)}")


def test_child_inherits_neither_launch_fd_nor_var(monkeypatch):
    br = _FakeBridge(_ok())
    fd = _pipe_with(SECRET.encode())
    monkeypatch.setenv("RELAY_LAUNCH_FD", str(fd))
    monkeypatch.setenv("RELAY_BRIDGE_SOCKET", br.path)
    monkeypatch.setenv("RELAY_SERVICE_ID", "relaytts-daemon")
    try:
        assert relay_bridge.establish_launch_identity() is True
    finally:
        br.close()
    probe = ("import os, sys\n"
             f"try:\n    os.fstat({fd}); print('fd-open')\n"
             "except OSError:\n    print('fd-closed')\n"
             "print(os.environ.get('RELAY_LAUNCH_FD', 'unset'))\n")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ["fd-closed", "unset"]


def test_daemon_exits_nonzero_when_identity_fails(monkeypatch):
    fd = _pipe_with(b"not-a-secret")
    monkeypatch.setenv("RELAY_LAUNCH_FD", str(fd))
    monkeypatch.setenv("RELAY_BRIDGE_SOCKET", "/tmp/rtts-no-such-bridge.sock")
    monkeypatch.setenv("RELAY_SERVICE_ID", "relaytts-daemon")
    try:
        relaytts_daemon.establish_relay_identity()
    except SystemExit as e:
        assert e.code == 78
    else:
        raise AssertionError("daemon continued without its launch identity")


def test_bridge_disabled_without_bound_identity(monkeypatch):
    monkeypatch.setenv("RELAY_BRIDGE_SOCKET", "/tmp/whatever.sock")
    monkeypatch.setenv("RELAY_SERVICE_ID", "relaytts-daemon")
    b = relay_bridge.RelayBridge(lambda: {}, "/abs/v.json", ["a"], "/data/c",
                                 identity_bound=False)
    assert not b.enabled


def test_register_manifest_on_the_wire_carries_no_token(monkeypatch):
    br = _FakeBridge(lambda req: b'{"type":"OK"}\n')
    monkeypatch.setenv("RELAY_BRIDGE_SOCKET", br.path)
    monkeypatch.setenv("RELAY_SERVICE_ID", "relaytts-daemon")
    b = relay_bridge.RelayBridge(lambda: {}, "/abs/v.json", ["a"], "/data/c",
                                 identity_bound=True)
    try:
        b.start()
    finally:
        b.stop()
        br.close()
    assert len(br.requests) == 1
    req = br.requests[0]
    assert req["type"] == "RegisterManifest"
    assert "token" not in req
    assert len(req["arguments"]["internalToken"]) == 64


if __name__ == "__main__":
    # pytest is not in requirements.txt, so this file stays runnable without it.
    # With pytest present, defer to it — the fixture-based tests below only run
    # that way. Without it, run what can be driven by hand and say what was
    # skipped rather than reporting a clean sweep that skipped a third of them.
    try:
        import pytest as _pytest
    except ImportError:
        _pytest = None

    if _pytest is not None:
        sys.exit(_pytest.main([os.path.abspath(__file__), "-q"]))

    import inspect
    import pathlib
    import tempfile
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed = skipped = 0
    for name, fn in fns:
        params = inspect.signature(fn).parameters
        if "monkeypatch" in params:
            print(f"  SKIP {name} (needs pytest)")
            skipped += 1
            continue
        try:
            if params:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception:
            print(f"  FAIL {name}")
            traceback.print_exc()
    total = len(fns) - skipped
    print(f"\n{passed}/{total} passed, {skipped} skipped (install pytest to run all)")
    sys.exit(0 if passed == total else 1)


def test_ci_break_scratch():
    assert False, "deliberate CI break"
