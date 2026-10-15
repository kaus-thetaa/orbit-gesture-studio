"""Per-OS shortcut tables used by the window-management actions.

Window management is done by sending the operating system's own shortcuts rather than
poking at window handles - it is simple, auditable and uses native behaviour. The
tables are plain data; users can override entries in settings.json
("safety.platform_shortcuts"). A value of None means "not supported on this platform".
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Mapping, Optional

from .keyboard import ShortcutError, canonicalize

log = logging.getLogger(__name__)

WINDOWS = {
    "snap_left": "win+left",
    "snap_right": "win+right",
    "show_desktop": "win+d",
    "window_overview": "win+tab",
}
# GNOME-style defaults; other desktops differ. Needs an X11 session (pynput can't inject on Wayland).
LINUX = {
    "snap_left": "win+left",
    "snap_right": "win+right",
    "show_desktop": "win+d",
    "window_overview": None,
}
# macOS has no native half-screen snap or show-desktop *shortcut* that is enabled by default.
MACOS = {
    "snap_left": None,
    "snap_right": None,
    "show_desktop": "f11",
    "window_overview": "ctrl+up",
}


@dataclass(frozen=True)
class PlatformAdapter:
    name: str
    shortcuts: Mapping[str, Optional[str]]

    def shortcut_for(self, key: str) -> Optional[str]:
        return self.shortcuts.get(key)


def detect_platform(overrides: Optional[Mapping[str, str]] = None, platform: Optional[str] = None) -> PlatformAdapter:
    platform = platform or sys.platform
    if platform.startswith("win"):
        name, table = "windows", dict(WINDOWS)
    elif platform == "darwin":
        name, table = "macos", dict(MACOS)
    else:
        name, table = "linux", dict(LINUX)
    for key, value in (overrides or {}).items():
        if key not in table:
            log.warning("platform_shortcuts: unknown key %r ignored", key)
            continue
        if value is None or value == "":
            table[key] = None
            continue
        try:
            canonicalize(value)
        except ShortcutError as exc:
            log.warning("platform_shortcuts[%s]=%r ignored: %s", key, value, exc)
            continue
        table[key] = value
    return PlatformAdapter(name, table)
