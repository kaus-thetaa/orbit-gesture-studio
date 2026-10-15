"""Small filesystem helpers: atomic JSON writes and tolerant reads."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON so a crash mid-write can never leave a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def quarantine_corrupt(path: Path) -> Path:
    """Move an unreadable file aside (never delete user data) and return the new path."""
    path = Path(path)
    target = path.with_suffix(path.suffix + ".corrupt")
    n = 1
    while target.exists():
        target = path.with_suffix(f"{path.suffix}.corrupt{n}")
        n += 1
    os.replace(path, target)
    log.warning("Moved unreadable file %s -> %s", path, target)
    return target
