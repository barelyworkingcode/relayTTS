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
        # `service register` is the only way to set env; there is no update
        # verb. So say so rather than silently ignoring a changed URL.
        if [ -n "${RELAYTTS_REMOTE_URL:-}" ] || [ -n "${RELAYTTS_REMOTE_MODEL:-}" ]; then
            echo "NOTE: a RELAYTTS_REMOTE_* variable is set but the service is" \
                 "already registered. To change it, unregister first:"
            echo "      $RELAY service unregister --name relaytts-daemon && ./build.sh"
        fi
    else
        # The daemon owns no model — RELAYTTS_REMOTE_URL / RELAYTTS_REMOTE_MODEL
        # are the deployment facts that point it at the server that does, and
        # `service register` is the only chance to bake them in. Fail loudly
        # here rather than register a daemon that will 500 on every request.
        MISSING=()
        [ -z "${RELAYTTS_REMOTE_URL:-}" ] && MISSING+=("RELAYTTS_REMOTE_URL")
        [ -z "${RELAYTTS_REMOTE_MODEL:-}" ] && MISSING+=("RELAYTTS_REMOTE_MODEL")
        if [ ${#MISSING[@]} -gt 0 ]; then
            echo "Missing required env var(s) for registration: ${MISSING[*]}" >&2
            echo "  RELAYTTS_REMOTE_URL=https://<host>:<port>/v1 \\" >&2
            echo "  RELAYTTS_REMOTE_MODEL=<id-that-server-exposes> ./build.sh" >&2
            exit 1
        fi

        REGISTER_ENV=(
            --env "RELAYTTS_REMOTE_URL=$RELAYTTS_REMOTE_URL"
            --env "RELAYTTS_REMOTE_MODEL=$RELAYTTS_REMOTE_MODEL"
        )
        if [ -n "${RELAYTTS_REMOTE_CLONE_MODEL:-}" ]; then
            REGISTER_ENV+=(--env "RELAYTTS_REMOTE_CLONE_MODEL=$RELAYTTS_REMOTE_CLONE_MODEL")
        fi
        if [ -n "${RELAYTTS_REMOTE_CA:-}" ]; then
            REGISTER_ENV+=(--env "RELAYTTS_REMOTE_CA=$RELAYTTS_REMOTE_CA")
        fi
        if [ -n "${RELAYTTS_REMOTE_PIN_SHA256:-}" ]; then
            REGISTER_ENV+=(--env "RELAYTTS_REMOTE_PIN_SHA256=$RELAYTTS_REMOTE_PIN_SHA256")
        fi

        "$RELAY" service register \
            --name relaytts-daemon \
            --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" \
            --autostart \
            --no-frontend-creds \
            "${REGISTER_ENV[@]}"
        echo "Registered relaytts-daemon service with Relay (remote: $RELAYTTS_REMOTE_URL)"
    fi
else
    echo "Relay not found at $RELAY, skipping registration"
fi
