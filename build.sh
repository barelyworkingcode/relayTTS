#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure conda environment exists
if ! conda info --envs 2>/dev/null | grep -q "^relaytts "; then
    echo "Setting up conda environment..."
    "$SCRIPT_DIR/setup_env.sh"
fi

# Register with Relay (best-effort)
RELAY="/Applications/Relay.app/Contents/MacOS/relay"
if [ -x "$RELAY" ]; then
    # Unregister kokoro-daemon if it's still registered — relayTTS is the
    # drop-in replacement and the two daemons must not both bind port 9997.
    if "$RELAY" service list 2>/dev/null | grep -q "kokoro-daemon"; then
        echo "Unregistering kokoro-daemon (replaced by relaytts-daemon)..."
        "$RELAY" service unregister --name kokoro-daemon
        echo "kokoro-daemon unregistered"
    fi

    # Remote inference is a deployment choice, not a source change: export
    # RELAYTTS_REMOTE_URL before running this and it is baked into the service
    # registration, so the daemon comes up in remote mode and loads no model.
    #
    #   RELAYTTS_REMOTE_URL=http://<router>:<port>/v1 \\
    #   RELAYTTS_REMOTE_MODEL=<id-that-server-exposes> ./build.sh
    #
    REGISTER_ENV=()
    if [ -n "${RELAYTTS_REMOTE_URL:-}" ]; then
        REGISTER_ENV+=(--env "RELAYTTS_REMOTE_URL=$RELAYTTS_REMOTE_URL")
    fi
    if [ -n "${RELAYTTS_REMOTE_MODEL:-}" ]; then
        REGISTER_ENV+=(--env "RELAYTTS_REMOTE_MODEL=$RELAYTTS_REMOTE_MODEL")
    fi
    if [ -n "${RELAYTTS_REMOTE_CLONE_MODEL:-}" ]; then
        REGISTER_ENV+=(--env "RELAYTTS_REMOTE_CLONE_MODEL=$RELAYTTS_REMOTE_CLONE_MODEL")
    fi

    if "$RELAY" service list 2>/dev/null | grep -q "relaytts-daemon"; then
        echo "Already registered with Relay. Daemon will use updated scripts."
        # `service register` is the only way to set env; there is no update
        # verb. So say so rather than silently ignoring a changed URL.
        if [ ${#REGISTER_ENV[@]} -gt 0 ]; then
            echo "NOTE: RELAYTTS_REMOTE_URL is set but the service is already" \
                 "registered. To change it, unregister first:"
            echo "      $RELAY service unregister --name relaytts-daemon && ./build.sh"
        fi
    else
        "$RELAY" service register \
            --name relaytts-daemon \
            --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" \
            --autostart \
            --no-frontend-creds \
            "${REGISTER_ENV[@]+"${REGISTER_ENV[@]}"}"
        if [ ${#REGISTER_ENV[@]} -gt 0 ]; then
            echo "Registered relaytts-daemon service with Relay (remote: $RELAYTTS_REMOTE_URL)"
        else
            echo "Registered relaytts-daemon service with Relay (local model)"
        fi
    fi
else
    echo "Relay not found at $RELAY, skipping registration"
fi
