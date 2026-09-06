#!/usr/bin/env python3
"""Fail-closed TLS with certificate pinning for the remote TTS transport.

No skip-verify / insecure-mode escape hatch exists here, and none should be
added: a knob to bypass verification is what turns a pinned transport into an
unpinned one silently, the exact failure mode this module exists to prevent.

Shares its shape with the equivalent module in the sibling relaySTT repo
(same rules, same function names) but is a standalone copy — nothing here
imports across repos.
"""
import hashlib
import http.client
import os
import ssl
import urllib.parse
import urllib.request

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
_HEX_DIGITS = set("0123456789abcdef")


def _normalize_pin(pin: str) -> str:
    """Strip colons/whitespace, lowercase, and require a 64-hex-char SHA-256."""
    normalized = str(pin).strip().replace(":", "").lower()
    if len(normalized) != 64 or any(c not in _HEX_DIGITS for c in normalized):
        raise ValueError(
            f"pin_sha256 entry {pin!r} is not a 64-character hex SHA-256 "
            "fingerprint (colons optional, case-insensitive)")
    return normalized


def parse_pins(text_or_list) -> list:
    """Normalize RELAYTTS_REMOTE_PIN_SHA256 / config `pin_sha256` into a list
    of lowercase hex fingerprints. Accepts a comma-separated string or a list;
    a blank string or None yields no pins."""
    if not text_or_list:
        return []
    items = text_or_list.split(",") if isinstance(text_or_list, str) else list(text_or_list)
    return [_normalize_pin(p) for p in items if str(p).strip()]


def assert_transport_config(url: str, ca_file, pins) -> None:
    """Validate a remote TTS endpoint before it is ever dialed.

    Raises ValueError naming exactly what the operator must change. Called
    from RemoteEngine.__init__, so a misconfigured endpoint fails the daemon
    at startup rather than on the first request.
    """
    if not url:
        raise ValueError(
            "engine.remote.base_url is required: set RELAYTTS_REMOTE_URL or "
            "engine.remote.base_url in config.yaml")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(
            f"engine.remote.base_url {url!r} must be http or https, got scheme {parts.scheme!r}")

    pins = [_normalize_pin(p) for p in (pins or [])]

    if parts.scheme == "http":
        if parts.hostname not in _LOOPBACK_HOSTS:
            raise ValueError(
                f"engine.remote.base_url {url!r} is plain http to a non-loopback host; "
                "switch to https and set RELAYTTS_REMOTE_CA (plus optionally "
                "RELAYTTS_REMOTE_PIN_SHA256) instead of sending the request text over the "
                "network unencrypted")
        if ca_file or pins:
            raise ValueError(
                "engine.remote.ca_file / pin_sha256 have no effect on a plain http URL "
                f"({url!r}); drop them or switch base_url to https — leaving them set would "
                "be a false sense of safety")
        return

    if ca_file:
        if not os.path.isfile(ca_file):
            raise ValueError(f"engine.remote.ca_file {ca_file!r} does not exist or is not a file")
        try:
            ssl.create_default_context(cafile=ca_file)
        except (OSError, ssl.SSLError) as e:
            raise ValueError(
                f"engine.remote.ca_file {ca_file!r} could not be loaded as a CA bundle: {e}") from None


def build_opener(url: str, ca_file, pins):
    """Build the urllib opener a RemoteEngine uses for every request.

    Plain http (loopback only — assert_transport_config enforces that) gets
    the stock opener. https gets a context anchored on `ca_file` when given
    (system roots are NOT consulted alongside it — ca_file is the only trust
    anchor) or the system roots otherwise; `pins`, when non-empty, additionally
    require the leaf certificate's SHA-256 fingerprint to match one of them,
    closing the connection if it does not.

    Never call urllib.request.install_opener with the result — this opener is
    scoped to one engine, not process-global.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        return urllib.request.build_opener()

    ctx = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED

    pins = set(pins or [])
    if not pins:
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))

    class _PinnedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            super().connect()
            der = self.sock.getpeercert(binary_form=True)
            fingerprint = hashlib.sha256(der).hexdigest()
            if fingerprint not in pins:
                self.sock.close()
                raise ssl.SSLCertVerificationError(
                    f"remote certificate fingerprint {fingerprint} is not pinned")

    class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(_PinnedHTTPSConnection, req, context=ctx)

    return urllib.request.build_opener(_PinnedHTTPSHandler())
