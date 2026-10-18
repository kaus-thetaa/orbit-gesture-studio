"""API keys live in the environment or a 0600 file in the config dir - never in settings.json or git."""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Mapping, Optional

from .. import paths
from ..fsutil import atomic_write_json

log = logging.getLogger(__name__)


def secrets_path() -> Path:
    return paths.config_dir() / "secrets.json"


def get_secret(name: str, env: Optional[Mapping[str, str]] = None, path: Optional[Path] = None) -> Optional[str]:
    env = os.environ if env is None else env
    if env.get(name):
        return env[name].strip() or None
    path = Path(path) if path else secrets_path()
    try:
        mode = path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            log.warning("%s is readable by other users; tightening permissions", path)
            os.chmod(path, 0o600)
        value = json.loads(path.read_text(encoding="utf-8")).get(name)
        return value.strip() if isinstance(value, str) and value.strip() else None
    except (OSError, ValueError):
        return None


def set_secret(name: str, value: str, path: Optional[Path] = None) -> Path:
    path = Path(path) if path else secrets_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    data[name] = value.strip()
    atomic_write_json(path, data)
    try:
        os.chmod(path, 0o600)
    except OSError:  # e.g. Windows: ACLs apply instead
        pass
    return path
