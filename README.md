# relayTTS

Qwen3-TTS daemon — a drop-in replacement for the Kokoro daemon. Same TCP port (9997), same length-prefixed JSON protocol. Eve needs no changes.

Model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit` via mlx-audio on Apple Silicon.

## Quick Start

```bash
./setup_env.sh          # Create conda env + install deps
./build.sh              # Unregisters kokoro-daemon, registers relaytts-daemon with Relay (autostart)
./test_speak.sh "Hello" # Test: generates speech and plays via afplay
```

## Architecture

```
Eve Web Chat ──WS──► Eve Backend ──TCP──► relayTTS Daemon (port 9997)
                         │                       │
                    tts-service.js          relaytts_daemon.py
                    (Node.js TCP client)    (MLX/Qwen3-TTS server)
                         │
                    base64 WAV ──WS──► Browser (Web Audio API)
```

The daemon runs as a Relay autostart service. Eve connects via TCP, sends text + voice ID, receives base64-encoded WAV audio.

Qwen3-TTS is an instruction-driven model: each voice has a `instruct` field in `config.yaml` that shapes its emotion and delivery style (e.g. "Speak in a warm, friendly tone"). The daemon injects the instruct when generating — no phonemizer or G2P step required.

## Voices

9 speakers from Qwen3-TTS CustomVoice:

| Speaker    | Gender | Language    |
|------------|--------|-------------|
| `ono_anna` | F      | Japanese (default) |
| `ryan`     | M      | English     |
| `aiden`    | M      | English     |
| `sohee`    | F      | Korean      |
| `serena`   | F      | English     |
| `vivian`   | F      | English     |
| `uncle_fu` | M      | Chinese     |
| `eric`     | M      | English     |
| `dylan`    | M      | English     |

Voices are defined in `config.yaml`. Each entry specifies the `speaker`, an `instruct` string for emotion/delivery, and optional fields (lang, gender). `voice_aliases` in `config.yaml` maps legacy Kokoro IDs (e.g. `af_heart`, `am_adam`) to relayTTS speakers, so existing clients need no changes.

```yaml
# config.yaml voice schema (excerpt)
voices:
  - id: serena
    name: Serena
    lang: en
    gender: f
    speaker: serena
    instruct: "Speak in a warm, friendly, and clear tone."

voice_aliases:
  af_heart: serena
  am_adam: ryan
```

## test_speak.sh

```bash
./test_speak.sh "Hello world"              # default voice (ono_anna)
./test_speak.sh "Good morning" serena      # specific voice
./test_speak.sh "Fast speech" ryan 1.5    # voice + speed
./test_speak.sh --voices                   # list all voices
PORT=9998 ./test_speak.sh "test"           # alternate port
```

## Protocol

TCP on port 9997. Length-prefixed JSON (4-byte big-endian header). Identical to the Kokoro protocol.

**Request:**
```json
{ "text": "Hello world", "voice": "serena", "speed": 1.0 }
```

**Response:**
```json
{
  "success": true,
  "audio_base64": "<base64 WAV>",
  "sample_rate": 24000,
  "duration": 1.5,
  "generation_time": 0.4,
  "rtf": 0.27
}
```

## Files

```
relayTTS/
├── README.md
├── build.sh                   # Unregisters kokoro-daemon, registers relaytts-daemon with Relay
├── setup_env.sh               # Creates conda env + installs deps
├── requirements.in            # Top-level deps (intent); edit to change a dep
├── requirements.txt           # Hash-pinned lockfile (generate with pip-compile; do not hand-edit)
├── config.yaml                # Voice definitions, instruct strings, voice_aliases
├── test_speak.sh              # CLI test tool
└── daemon/
    ├── relaytts_daemon.py     # TCP server, MLX/Qwen3-TTS model, WAV generation
    └── daemon_wrapper.sh      # Conda wrapper + restart-on-crash supervisor
```

## Dependencies

- macOS with Apple Silicon
- Conda (miniconda/miniforge)
- ffmpeg (`brew install ffmpeg`) — runtime system dep for pitch-preserving time-stretch
- Python packages: `mlx-audio>=0.4.4`, `mlx`, `soundfile`, `numpy`, `pyyaml`

No espeak-ng, misaki, phonemizer, or spaCy — Qwen3-TTS needs no G2P phonemizer step.

### Dependency management

Once a hash-pinned lockfile is needed:

```bash
pip-compile --generate-hashes --allow-unsafe --output-file requirements.txt requirements.in
```

Then reinstall with `setup_env.sh` (it prefers `requirements.txt` over `requirements.in` when both exist).
