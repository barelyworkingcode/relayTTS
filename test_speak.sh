#!/bin/bash
# Quick test: send text to relayTTS daemon and play the resulting audio
#
# Usage:
#   ./test_speak.sh "Hello, this is a test"
#   ./test_speak.sh "Hello" ryan               # specific voice
#   ./test_speak.sh "Hello" aiden 1.2          # voice + speed
#   ./test_speak.sh --voices                   # list available voices
#   PORT=9998 ./test_speak.sh "test"           # alternate port
set -euo pipefail

# --- Voice catalog ---
show_voices() {
    cat <<'VOICES'
Qwen3-TTS CustomVoice speakers (via mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit):

  ono_anna   — Japanese female (default)
  ryan       — English male
  aiden      — English male
  sohee      — Korean female
  serena     — English female
  vivian     — English female
  uncle_fu   — Chinese male
  eric       — English male
  dylan      — English male

Voices and per-voice instruct (emotion/delivery) are configured in config.yaml.
Legacy Kokoro voice IDs (e.g. af_heart, am_adam) are remapped via voice_aliases
in config.yaml so existing clients need no changes.
VOICES
}

if [[ "${1:-}" == "--voices" || "${1:-}" == "-v" ]]; then
    show_voices
    exit 0
fi

TEXT="${1:-Hello! This is a test of the relayTTS text to speech system.}"
VOICE="${2:-ono_anna}"
SPEED="${3:-1.0}"
PORT="${PORT:-9997}"
HOST="${HOST:-localhost}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/bin/activate" relaytts 2>/dev/null || true

python3 -c "
import socket, struct, json, base64, sys, tempfile, subprocess, os

host, port = '$HOST', $PORT
request = {'text': '''$TEXT''', 'voice': '$VOICE', 'speed': $SPEED}

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((host, port))
except ConnectionRefusedError:
    print(f'Cannot connect to relayTTS daemon at {host}:{port}')
    print('Start it with: ./build.sh  (or via Relay)')
    sys.exit(1)

payload = json.dumps(request).encode('utf-8')
sock.sendall(struct.pack('!I', len(payload)) + payload)

header = b''
while len(header) < 4:
    header += sock.recv(4 - len(header))
resp_len = struct.unpack('!I', header)[0]

data = b''
while len(data) < resp_len:
    data += sock.recv(min(resp_len - len(data), 65536))
sock.close()

response = json.loads(data.decode('utf-8'))

if not response.get('success'):
    print(f'Error: {response.get(\"error\", \"unknown\")}')
    sys.exit(1)

wav_data = base64.b64decode(response['audio_base64'])
duration = response.get('duration', '?')
rtf = response.get('rtf', '?')
gen_time = response.get('generation_time', '?')
print(f'Generated {duration}s audio in {gen_time}s (RTF: {rtf}x) voice={\"$VOICE\"}')

tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
tmp.write(wav_data)
tmp.close()

subprocess.run(['afplay', tmp.name])
os.unlink(tmp.name)
"
