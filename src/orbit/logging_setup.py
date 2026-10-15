"""Structured-ish logging: console + rotating file in the ORBIT config dir."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import Optional

from . import paths

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class RingBufferHandler(logging.Handler):
    """Keeps recent log lines in memory so the GUI can display them."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.capacity = capacity
        self.lines: list = []
        self.listeners: list = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # pragma: no cover - logging must never raise
            return
        self.lines.append(line)
        if len(self.lines) > self.capacity:
            del self.lines[: len(self.lines) - self.capacity]
        for cb in list(self.listeners):
            try:
                cb(line)
            except Exception:
                pass


ring = RingBufferHandler()


def setup_logging(level: str = "INFO", to_file: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        if getattr(h, "_orbit", False):
            root.removeHandler(h)

    fmt = logging.Formatter(_FMT)
    handlers: list = []
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    handlers.append(console)
    if to_file:
        try:
            fh = logging.handlers.RotatingFileHandler(
                paths.logs_dir() / "orbit.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            handlers.append(fh)
        except OSError as exc:
            print(f"warning: file logging disabled ({exc})", file=sys.stderr)
    ring.setFormatter(fmt)
    handlers.append(ring)
    for h in handlers:
        h._orbit = True  # type: ignore[attr-defined]
        root.addHandler(h)
