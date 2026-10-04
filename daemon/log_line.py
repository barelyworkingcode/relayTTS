"""One-JSON-object-per-line logging to stderr, per relay's logging standard.

Never logs a prompt, body, token or synthesized text: callers pass names and ids.
"""
import contextlib
import contextvars
import datetime
import ipaddress
import json
import os
import re
import secrets
import sys
import threading
import time
from urllib.parse import urlsplit

DEFAULT_SERVICE = "relaytts-daemon"
DEBUG_WINDOW_S = 30 * 60
_LEVELS = {"error": 0, "warn": 1, "info": 2, "debug": 3}
_TRACE_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
# Keys the writer owns; a caller attribute with one of these names is renamed.
_RESERVED = ("ts", "level", "msg", "service", "trace_id")
_CORE = ("op", "status", "duration_ms", "error")
_STATUSES = ("ok", "error", "denied")

_trace_var = contextvars.ContextVar("relaytts_trace_id", default="")


def new_trace_id() -> str:
    return secrets.token_hex(16)


def valid_trace_id(s) -> bool:
    return isinstance(s, str) and _TRACE_RE.fullmatch(s) is not None


def accept_trace_id(v) -> str:
    """Keep an inbound id if well-formed, else mint one. A rejected value is
    never logged: it could carry an injected line."""
    return v if valid_trace_id(v) else new_trace_id()


def current_trace_id() -> str:
    return _trace_var.get()


@contextlib.contextmanager
def trace_scope(trace_id):
    token = _trace_var.set(trace_id if valid_trace_id(trace_id) else "")
    try:
        yield
    finally:
        _trace_var.reset(token)


def is_on_box_url(url) -> bool:
    """True for unix: URLs and loopback hosts (localhost, 127.0.0.0/8, ::1)."""
    if not isinstance(url, str):
        return False
    if url.lower().startswith("unix:"):
        return True
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Logger:
    def __init__(self, stream=None, env=None, monotonic=time.monotonic, wall=time.time):
        env = os.environ if env is None else env
        level = (env.get("RELAY_LOG_LEVEL") or "").strip().lower()
        self._level = level if level in _LEVELS else "info"
        self.service = env.get("RELAY_SERVICE_ID") or DEFAULT_SERVICE
        self._stream = stream
        self._monotonic = monotonic
        self._wall = wall
        self._start = monotonic() if self._level == "debug" else None
        self._lock = threading.Lock()

    def _ts(self) -> str:
        t = self._wall()
        dt = datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

    def _write(self, record) -> None:
        try:
            line = json.dumps(record, default=str, ensure_ascii=True)
            stream = self._stream if self._stream is not None else sys.stderr
            stream.write(line + "\n")
            stream.flush()
        except Exception:
            pass

    def _build(self, level, msg, op, status, duration_ms, error, trace_id, attrs):
        if status not in _STATUSES:
            status = "error" if level in ("error", "warn") else "ok"
        try:
            duration = max(0, int(duration_ms))
        except (TypeError, ValueError):
            duration = 0
        record = {
            "ts": self._ts(),
            "level": level,
            "msg": str(msg)[:500],
            "service": self.service,
            "op": op,
            "status": status,
            "duration_ms": duration,
            "error": str(error or "")[:500],
            "trace_id": trace_id,
        }
        for k, v in attrs.items():
            if k == "status":
                k = "http_status"
            elif k in _RESERVED or k in _CORE:
                k = "attr_" + k
            record[k] = v
        return record

    def log(self, level, msg, /, *, op="log", status=None, duration_ms=0, error="",
            **attrs):
        try:
            if self._start is not None and \
                    self._monotonic() - self._start >= DEBUG_WINDOW_S:
                with self._lock:
                    expired = self._start is not None
                    if expired:
                        self._start = None
                        self._level = "info"
                if expired:
                    self._write(self._build(
                        "warn", "debug logging ended after 30 minutes", "log",
                        "error", 0, "", "", {}))
            if level not in _LEVELS or _LEVELS[level] > _LEVELS[self._level]:
                return
            # The writer alone owns trace_id: it comes from the context.
            trace_id = current_trace_id()
            if status is not None and not isinstance(status, str):
                # A numeric status is an HTTP code, not the outcome.
                attrs.setdefault("http_status", status)
                status = None
            if status is None:
                status = "error" if level in ("error", "warn") else "ok"
            self._write(self._build(level, msg, op, status, duration_ms, error,
                                    trace_id, attrs))
        except Exception:
            pass

    def debug(self, msg, /, **kw):
        self.log("debug", msg, **kw)

    def info(self, msg, /, **kw):
        self.log("info", msg, **kw)

    def warn(self, msg, /, **kw):
        self.log("warn", msg, **kw)

    def error(self, msg, /, **kw):
        self.log("error", msg, **kw)


logger = Logger()


def log(level, msg, /, **kw):
    logger.log(level, msg, **kw)


def debug(msg, /, **kw):
    logger.log("debug", msg, **kw)


def info(msg, /, **kw):
    logger.log("info", msg, **kw)


def warn(msg, /, **kw):
    logger.log("warn", msg, **kw)


def error(msg, /, **kw):
    logger.log("error", msg, **kw)
