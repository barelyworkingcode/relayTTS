#!/usr/bin/env python3
"""Unit tests for the parts of the daemon that don't need the model: custom
voice loading/merge/resolve precedence, and the relay manifest/registration
framing. Runs under pytest, or standalone (`python test_relaytts.py`).

Importing relaytts_daemon pulls in numpy/soundfile/yaml (present in the relaytts
env) but NOT mlx-audio — the model is loaded lazily inside load_model(), so
these stay fast and model-free.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import relay_bridge
from relaytts_daemon import Config

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


def _make_config(tmp_dir, custom=None, clones=None):
    cfg_path = os.path.join(tmp_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(CONFIG_YAML)
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


if __name__ == "__main__":
    import tempfile, traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed = 0
    for name, fn in fns:
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as d:
                    fn(d)
            else:
                fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception:
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
