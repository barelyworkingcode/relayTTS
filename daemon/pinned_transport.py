#!/usr/bin/env python3
"""Fail-closed TLS with certificate pinning for the remote TTS transport, plus
the `unix:<path>` transport used when the remote is relay's own model.sock.

No skip-verify / insecure-mode escape hatch exists here, and none should be
added: a knob to bypass verification is what turns a pinned transport into an
unpinned one silently, the exact failure mode this module exists to prevent.
The unix transport has no analogous knob either — there is nothing to pin,
since relay identifies this service by its launch identity, not a
certificate (see `UnixSocketOpener`).

Shares its shape with the equivalent module in the sibling relaySTT repo
(same rules, same function names) but is a standalone copy — nothing here
imports across repos.
"""
import hashlib
import http.client
import io
import os
import socket
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
_HEX_DIGITS = set("0123456789abcdef")
_UNIX_PREFIX = "unix:"


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


def is_unix_url(url: str) -> bool:
    """True for the `unix:<abs path>` form of a remote base_url — HTTP over
    AF_UNIX rather than TCP, with no bearer header ever sent (relay
    identifies the caller by its launch identity; see
    relay/docs/model-endpoint.md's Auth order)."""
    return bool(url) and url.startswith(_UNIX_PREFIX)


def parse_unix_socket_path(url: str) -> str:
    """Extract and validate the socket path from a `unix:<path>` base_url.

    The path is handed straight to socket.connect(), never joined against a
    directory, so a relative path would silently resolve against whatever
    the daemon's CWD happens to be at connect time rather than the path the
    operator wrote — refused here instead, at startup.
    """
    path = url[len(_UNIX_PREFIX):]
    if not path.startswith("/"):
        raise ValueError(
            f"engine.remote.base_url {url!r} names a unix socket with a relative "
            "path; use an absolute path, e.g. unix:/path/to/model.sock")
    return path


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

    if is_unix_url(url):
        parse_unix_socket_path(url)
        pins = [_normalize_pin(p) for p in (pins or [])]
        if ca_file or pins:
            raise ValueError(
                f"engine.remote.ca_file / pin_sha256 have no effect on a unix "
                f"socket base_url ({url!r}); relay identifies this service by "
                "its launch identity, not TLS — drop them")
        return

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
    """Build the opener a RemoteEngine uses for every request.

    `unix:<path>` gets a `UnixSocketOpener` — HTTP over AF_UNIX, no TLS and
    no bearer header (see that class). Plain http (loopback only —
    assert_transport_config enforces that) gets the stock opener. https gets
    a context anchored on `ca_file` when given (system roots are NOT
    consulted alongside it — ca_file is the only trust anchor) or the system
    roots otherwise; `pins`, when non-empty, additionally require the leaf
    certificate's SHA-256 fingerprint to match one of them, closing the
    connection if it does not.

    Never call urllib.request.install_opener with the result — this opener is
    scoped to one engine, not process-global.
    """
    if is_unix_url(url):
        return UnixSocketOpener(parse_unix_socket_path(url))

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


# ── unix:<path> transport: HTTP over AF_UNIX, no TLS, no bearer header ──
#
# Used when the remote is relay's own model.sock: relay identifies the caller
# by the kernel's audit token on the connection, not by anything in the
# request, so this transport never has a credential to attach in the first
# place — see relay/docs/model-endpoint.md's Auth order.

class _UnixHTTPConnection(http.client.HTTPConnection):
    """http.client.HTTPConnection dialed over AF_UNIX instead of AF_INET.

    `host` is a fixed placeholder — relay's model.sock does not
    hostname-route, but HTTP/1.1 still wants a Host header, and http.client
    derives one from `self.host`.
    """

    def __init__(self, sock_path: str, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        super().__init__("model.sock", timeout=timeout)
        self._sock_path = sock_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            sock.settimeout(self.timeout)
        sock.connect(self._sock_path)
        self.sock = sock


class _BufferedResponse:
    """A fully-read response, matching the subset of urllib's response
    interface RemoteEngine uses (`read()`, used as a context manager). The
    body is read out under UnixSocketOpener's lock before this is handed
    back, so the caller can take as long as it likes with it without holding
    the connection open for anyone else — see UnixSocketOpener.open."""

    def __init__(self, status: int, body: bytes):
        self.status = self.code = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class UnixSocketOpener:
    """RemoteEngine's transport for a `unix:<path>` base_url: HTTP/1.1 over
    AF_UNIX, one kept-alive connection reused across requests rather than one
    dialed and torn down per call — the file-descriptor-leak failure mode a
    per-call transport has if nothing ever closes it (see
    plan-broker-and-sessions.md's R-M1b security-review amendment on this
    exact shape, upstream side).

    A single connection, not a pool: `_lock` serializes every request,
    because RelayTTSDaemon dispatches one thread per TCP client and
    http.client connections handle one in-flight request at a time.
    Synthesis calls are already effectively serialized against a single
    remote model, so this costs nothing a pool would have avoided. A
    connection the peer has since idle-closed — relay is an ordinary
    HTTP/1.1 server about that — is detected and replaced once before
    giving up.
    """

    def __init__(self, sock_path: str):
        self._sock_path = sock_path
        self._lock = threading.Lock()
        self._conn: _UnixHTTPConnection | None = None

    def _drop_connection(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def open(self, req, timeout=None):
        """Send `req` (a urllib.request.Request built against a `unix://`
        URL) and return a `_BufferedResponse`, or raise `urllib.error.
        HTTPError` / `URLError` — the same exceptions RemoteEngine already
        handles for the http/https transports."""
        with self._lock:
            headers = dict(req.header_items())
            selector = req.selector
            method = req.get_method()
            body = req.data

            last_exc = None
            status = reason = resp_headers = will_close = data = None
            for attempt in range(2):
                if self._conn is None:
                    self._conn = _UnixHTTPConnection(self._sock_path, timeout=timeout)
                conn = self._conn
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
                try:
                    conn.request(method, selector, body=body, headers=headers)
                    resp = conn.getresponse()
                    data = resp.read()
                    status, reason = resp.status, resp.reason
                    resp_headers, will_close = resp.headers, resp.will_close
                    break
                except (http.client.BadStatusLine, http.client.RemoteDisconnected,
                        ConnectionError, OSError) as e:
                    self._drop_connection()
                    last_exc = e
            else:
                raise urllib.error.URLError(last_exc)

            if will_close:
                self._drop_connection()

        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, reason, resp_headers,
                                         io.BytesIO(data))
        return _BufferedResponse(status, data)
