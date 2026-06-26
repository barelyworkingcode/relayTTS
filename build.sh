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

    if "$RELAY" service list 2>/dev/null | grep -q "relaytts-daemon"; then
        echo "Already registered with Relay. Daemon will use updated scripts."
    else
        "$RELAY" service register \
            --name relaytts-daemon \
            --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" \
            --autostart \
            --no-frontend-creds
        echo "Registered relaytts-daemon service with Relay"
    fi
else
    echo "Relay not found at $RELAY, skipping registration"
fi
