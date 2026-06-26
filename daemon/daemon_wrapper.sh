#!/bin/bash
# Wrapper that activates the conda env and supervises the relayTTS daemon.

# Activate conda environment
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/bin/activate" relaytts

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Restart-on-crash supervision.
#
# The daemon wraps non-thread-safe native libs (MLX/Metal), which it drives from
# a single dedicated generation thread so concurrent requests can't crash it,
# but if the process ever still dies (OOM, an unrelated native fault) we restart
# it so
# TTS self-heals instead of staying dead until the next Relay launch — the
# daemon is registered --autostart only, with no restart-on-crash from Relay.
#
# Relay runs this wrapper as a process-group leader and stops the service by
# SIGTERM-ing the whole group (1s grace, then SIGKILL). So: run python in the
# background, wait on it, and trap TERM/INT to forward the signal, stop the loop,
# and exit promptly — well inside Relay's grace window. Only an unexpected exit
# (crash) triggers a respawn; a clean exit or a stop signal ends the loop.
term=0
child=""
shutdown() { term=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; }
trap shutdown TERM INT

while true; do
    python "$SCRIPT_DIR/relaytts_daemon.py" --idle-timeout 0 "$@" &
    child=$!
    wait "$child"
    code=$?
    if [ "$term" -eq 1 ] || [ "$code" -eq 0 ]; then
        break
    fi
    echo "relaytts daemon exited (code $code) — restarting in 2s" >&2
    sleep 2
done
