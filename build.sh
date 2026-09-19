#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure conda environment exists
CONDA_ENVS="$(conda info --envs 2>/dev/null || true)"
if ! grep -q "^relaytts " <<< "$CONDA_ENVS"; then
    echo "Setting up conda environment..."
    "$SCRIPT_DIR/setup_env.sh"
fi

# Register with Relay (best-effort)
RELAY="/Applications/Relay.app/Contents/MacOS/relay"
if [ -x "$RELAY" ]; then
    # Unregister kokoro-daemon if it's still registered — relayTTS is the
    # drop-in replacement and the two daemons must not both bind port 9997.
    # Capture before grep: `service list | grep -q` under pipefail exits 141
    # (SIGPIPE) when grep exits on a match before the list finishes writing.
    SERVICE_LIST="$("$RELAY" service list 2>/dev/null || true)"
    if grep -q "kokoro-daemon" <<< "$SERVICE_LIST"; then
        echo "Unregistering kokoro-daemon (replaced by relaytts-daemon)..."
        "$RELAY" service unregister --name kokoro-daemon
        echo "kokoro-daemon unregistered"
    fi

    # With relay present, its own model.sock is the default remote endpoint:
    # the daemon reaches its TTS model through relay's launch-identity auth
    # (no bearer header — see daemon/pinned_transport.py's unix transport)
    # instead of a directly-configured HTTPS endpoint. An explicit
    # RELAYTTS_REMOTE_URL (e.g. talking to a server directly, or in dev)
    # always wins.
    if [ -z "${RELAYTTS_REMOTE_URL:-}" ]; then
        RELAYTTS_REMOTE_URL="unix:$HOME/Library/Application Support/relay/model.sock"
    fi

    if grep -q "relaytts-daemon" <<< "$SERVICE_LIST"; then
        echo "Already registered with Relay. Daemon will use updated scripts."
        # `service register` is the only way to set env or capabilities;
        # there is no update verb — it upserts and restates the whole
        # record, so re-running with new values here has no effect on an
        # already-registered service. Say so rather than silently ignoring
        # a changed URL, model, or the new `models` capability.
        echo "NOTE: relay service register has no update verb. To pick up a" \
             "changed RELAYTTS_REMOTE_* variable, or the models capability /" \
             "allowed-model grant this build.sh now registers, unregister" \
             "first:"
        echo "      $RELAY service unregister --name relaytts-daemon && ./build.sh"
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

        # `models` scopes this service to exactly the model(s) named below
        # (relay/docs/model-endpoint.md: empty allowed_models means NO
        # models for a service, the opposite of a project's default) — TTS
        # is meant to reach its one remote model, not every model relay's
        # broker can reach. Only grant it when the daemon actually talks to
        # relay's model.sock: a plain https:// remote never calls relay's
        # model endpoint, so granting `models` there would be an unused,
        # unrevoked capability sitting on the service record.
        REGISTER_CAPS=(--capability manifest)
        case "$RELAYTTS_REMOTE_URL" in
            unix:*)
                REGISTER_CAPS+=(--capability models --allowed-model "$RELAYTTS_REMOTE_MODEL")
                if [ -n "${RELAYTTS_REMOTE_CLONE_MODEL:-}" ]; then
                    REGISTER_CAPS+=(--allowed-model "$RELAYTTS_REMOTE_CLONE_MODEL")
                fi
                ;;
        esac

        "$RELAY" service register \
            --name relaytts-daemon \
            --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" \
            --autostart \
            "${REGISTER_CAPS[@]}" \
            "${REGISTER_ENV[@]}"
        echo "Registered relaytts-daemon service with Relay (remote: $RELAYTTS_REMOTE_URL)"
    fi
else
    echo "Relay not found at $RELAY, skipping registration"
fi
