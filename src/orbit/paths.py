"""Where ORBIT keeps user data. Always OUTSIDE the git repository.

Override with the ORBIT_HOME environment variable (handy for tests / portable use).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def config_dir() -> Path:
    env = os.environ.get("ORBIT_HOME")
    if env:
        p = Path(env).expanduser()
    elif sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        p = Path(base) / "ORBIT Gesture Studio"
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / "ORBIT Gesture Studio"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        p = Path(base) / "orbit-gesture-studio"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = config_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = config_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def settings_path() -> Path:
    return config_dir() / "settings.json"


def mappings_path() -> Path:
    return config_dir() / "mappings.json"


def custom_gestures_path() -> Path:
    return config_dir() / "custom_gestures.json"
