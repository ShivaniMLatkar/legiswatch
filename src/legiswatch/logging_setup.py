"""Structured JSON logging.

Compliance automation has to be auditable: when someone asks six months from
now why obligation X was auto-filed, the answer has to come out of a log, not
a memory. Every log line is a single JSON object with a stable set of keys.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def configure(level: str | None = None) -> None:
    """Idempotent for handlers, but an explicit level always wins.

    Modules call get_logger() at import time, which configures at the default
    level; a later explicit configure() from a CLI entry point must still be
    able to quiet things down.
    """
    global _configured
    if _configured:
        if level:
            logging.getLogger("legiswatch").setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger("legiswatch")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level or os.getenv("LEGISWATCH_LOG_LEVEL", "INFO"))
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure()
    short = name.split(".")[-1]
    return logging.getLogger(f"legiswatch.{short}")


@contextmanager
def timed(logger: logging.Logger, event: str, **fields: Any):
    """Emit a duration_ms for any block. Feeds the per-stage timings on the dashboard."""
    t0 = time.perf_counter()
    sink: dict[str, float] = {}
    try:
        yield sink
    finally:
        ms = (time.perf_counter() - t0) * 1000
        sink["duration_ms"] = ms
        logger.info(event, extra={"duration_ms": round(ms, 1), **fields})
