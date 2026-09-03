#!/usr/bin/env python3
"""Unit tests for the parts of the daemon that don't need the model: custom
voice loading/merge/resolve precedence, and the relay manifest/registration
framing. Runs under pytest, or standalone (`python test_relaytts.py`).

Importing relaytts_daemon pulls in numpy/soundfile/yaml (present in the relaytts
env) but NOT mlx-audio — the model is loaded lazily inside load_model(), so
these stay fast and model-free.
"""
import base64
import io
import json
import os
import sys
import urllib.error

try:
    import pytest
except ImportError:  # the fallback runner at the bottom covers this
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import relay_bridge
import relaytts_daemon
from relaytts_daemon import Config, RemoteEngine

CONFIG_YAML = """
engine:
  repo_id: test/model
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


REMOTE_BLOCK = """
  remote:
    enabled: true
    base_url: http://198.51.100.10:8080/v1/
    model: upstream/custom-voice
    clone_model: upstream/base
"""


def _make_config(tmp_dir, custom=None, clones=None, remote=None):
    cfg_path = os.path.join(tmp_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        # The remote block belongs under `engine:`, which CONFIG_YAML ends at
        # `temperature`, so appending there keeps the indentation right.
        f.write(CONFIG_YAML.replace("  temperature: 0.9\n",
                                    "  temperature: 0.9\n" + (remote or "")))
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
         "ref_audio": "/some/ref.wav", "ref_text": "Hello there.", "gain": 1.1, "speed": 1.0},
    ])
    assert cfg.voice_counts() == {"builtin": 2, "custom": 0, "clone": 1, "total": 3}
    spec = cfg.resolve("my_voice")
    assert spec == {"kind": "clone", "ref_audio": "/some/ref.wav",
                    "ref_text": "Hello there.", "gain": 1.1, "speed": 1.0}
    assert [v["id"] for v in cfg.public_voices()] == ["anna", "ryan", "my_voice"]
    # clone speakers must NOT pollute by_speaker (no "speaker" key on clones)
    assert cfg.resolve("ono_anna")["kind"] == "preset"


def test_clone_requires_ref_audio_and_text(tmp_path):
    cfg = _make_config(str(tmp_path), clones=[
        {"id": "no_audio", "ref_text": "hi"},                 # missing ref_audio
        {"id": "no_text", "ref_audio": "/x.wav"},             # missing ref_text
        {"ref_audio": "/y.wav", "ref_text": "hi"},            # missing id
        {"id": "ok", "ref_audio": "/z.wav", "ref_text": "ok"},
    ])
    assert cfg.voice_counts()["clone"] == 1
    assert cfg.resolve("ok")["ref_audio"] == "/z.wav"


def test_custom_and_clone_coexist(tmp_path):
    cfg = _make_config(
        str(tmp_path),
        custom=[{"id": "narrator", "base_speaker": "ryan", "instruct": "Slow."}],
        clones=[{"id": "cloned", "ref_audio": "/r.wav", "ref_text": "hi"}],
    )
    assert cfg.voice_counts() == {"builtin": 2, "custom": 1, "clone": 1, "total": 4}


def test_build_clone_schema_shape():
    schema = relay_bridge.build_clone_schema()
    assert schema[0]["id"] == "clones" and schema[0]["type"] == "array"
    fields = {f["id"]: f for f in schema[0]["item"]["fields"]}
    assert fields["ref_audio"]["required"] is True
    assert fields["ref_text"]["type"] == "textarea"
    assert "base_speaker" not in fields and "instruct" not in fields  # not for clones


def test_manifest_includes_both_arrays():
    m = relay_bridge.build_manifest("svc", "/abs/v.json", ["a"])
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
    m = relay_bridge.build_manifest("relaytts-daemon", "/abs/voices.json", ["a"])
    assert m["routes"] == ["/api/relaytts-daemon/"]   # non-empty (relay requires it)
    assert m["status"]["path"] == "/api/status"
    assert m["config"]["path"] == "/abs/voices.json"
    assert m["config"]["format"] == "json"
    assert m["config"]["applyMode"] == "live"


def test_register_payload_framing():
    m = relay_bridge.build_manifest("svc", "/abs/v.json", ["a"])
    raw = relay_bridge.build_register_payload("svc", m, "/tmp/i.sock", "itok", "stok")
    assert raw.endswith(b"\n")
    msg = json.loads(raw)
    assert msg["type"] == "RegisterManifest"
    assert msg["token"] == "stok"
    args = msg["arguments"]
    assert args["serviceId"] == "svc"
    assert args["internalSocket"] == "/tmp/i.sock"
    assert args["internalToken"] == "itok"
    assert args["manifest"]["status"]["path"] == "/api/status"
# ── Remote engine ─────────────────────────────────────────────────

def test_remote_disabled_by_default(tmp_path):
    cfg = _make_config(str(tmp_path))
    assert cfg.remote.enabled is False


def test_remote_enabled_from_config(tmp_path):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    assert cfg.remote.enabled is True
    assert cfg.remote.model == "upstream/custom-voice"
    assert cfg.remote.clone_model == "upstream/base"
    # Trailing slash on base_url must not double up in the joined path.
    assert cfg.remote.speech_url == "http://198.51.100.10:8080/v1/audio/speech"


def test_remote_env_url_enables_and_overrides(tmp_path, monkeypatch):
    """RELAYTTS_REMOTE_URL alone flips a deployment to remote, so a supervisor
    can do it without editing a tracked config file."""
    monkeypatch.setenv("RELAYTTS_REMOTE_URL", "http://host:9999/v1")
    cfg = _make_config(str(tmp_path), remote="""
  remote:
    enabled: false
    base_url: http://ignored:1/v1
    model: upstream/custom-voice
""")
    assert cfg.remote.enabled is True
    assert cfg.remote.speech_url == "http://host:9999/v1/audio/speech"


def test_remote_enabled_without_model_is_fatal(tmp_path):
    """Failing at load is the point: a remote daemon with no model id would
    otherwise start clean and 400 on every synthesis."""
    try:
        _make_config(str(tmp_path), remote="""
  remote:
    enabled: true
    base_url: http://198.51.100.10:8080/v1
""")
    except ValueError as e:
        assert "engine.remote.model" in str(e)
    else:
        raise AssertionError("expected ValueError for remote without a model")


def test_remote_clone_model_falls_back_to_model(tmp_path):
    cfg = _make_config(str(tmp_path), remote="""
  remote:
    enabled: true
    base_url: http://198.51.100.10:8080/v1
    model: upstream/only
""")
    assert cfg.remote.clone_model == "upstream/only"


def test_remote_api_key_read_from_named_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TTS_TOKEN", "s3cret")
    cfg = _make_config(str(tmp_path), remote="""
  remote:
    enabled: true
    base_url: http://198.51.100.10:8080/v1
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


def _capture_request(monkeypatch, body=None):
    """Intercept urlopen and hand back the Request the engine built."""
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["payload"] = json.loads(req.data.decode())
        seen["timeout"] = timeout
        return _FakeResponse(body if body is not None else _wav_bytes())

    monkeypatch.setattr(relaytts_daemon.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_remote_preset_payload_shape(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    seen = _capture_request(monkeypatch)
    spec = cfg.resolve("ryan")

    audio = cfg.remote.synthesize(spec, "Hello there.", "english",
                                  "Confident.", 0.9, 24000)

    assert seen["url"] == "http://198.51.100.10:8080/v1/audio/speech"
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
    seen = _capture_request(monkeypatch)
    cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert "instructions" not in seen["payload"]


def test_remote_clone_sends_base64_reference(tmp_path, monkeypatch):
    """The reference recording lives beside the daemon, so it has to travel
    with the request — the remote server has no access to that path."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(_wav_bytes(0.25))
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK, clones=[
        {"id": "mine", "name": "Mine", "ref_audio": str(ref), "ref_text": "A sample."}])
    seen = _capture_request(monkeypatch)

    cfg.remote.synthesize(cfg.resolve("mine"), "Hello.", "english", None, 0.9, 24000)

    assert seen["payload"]["model"] == "upstream/base"
    assert seen["payload"]["ref_text"] == "A sample."
    assert base64.b64decode(seen["payload"]["ref_audio"]) == ref.read_bytes()
    assert "voice" not in seen["payload"]


def test_remote_sends_bearer_only_when_configured(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    seen = _capture_request(monkeypatch)
    cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert not any(k.lower() == "authorization" for k in seen["headers"])

    monkeypatch.setenv("RELAYTTS_REMOTE_API_KEY", "tok")
    cfg2 = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    seen2 = _capture_request(monkeypatch)
    cfg2.remote.synthesize(cfg2.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert seen2["headers"]["Authorization"] == "Bearer tok"


def test_remote_http_error_names_endpoint_and_reason(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error":{"message":"unknown model"}}'))

    monkeypatch.setattr(relaytts_daemon.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as e:
        cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)
    assert "198.51.100.10:8080" in str(e.value)
    assert "HTTP 400" in str(e.value) and "unknown model" in str(e.value)


def test_remote_unreachable_error_is_actionable(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)

    def boom(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(relaytts_daemon.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="unreachable"):
        cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)


def test_remote_undecodable_body_is_reported(tmp_path, monkeypatch):
    cfg = _make_config(str(tmp_path), remote=REMOTE_BLOCK)
    _capture_request(monkeypatch, body=b"<html>proxy error</html>")
    with pytest.raises(RuntimeError, match="not decodable audio"):
        cfg.remote.synthesize(cfg.resolve("ryan"), "Hi.", "english", None, 0.9, 24000)


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
