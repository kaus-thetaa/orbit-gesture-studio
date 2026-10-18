"""Start ORBIT's orb at login (Linux/XDG autostart). Writes ONE small .desktop file; nothing else."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

NAME = "orbit-gesture-studio.desktop"


def autostart_path(home: Optional[Path] = None) -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path(home or Path.home()) / ".config"))
    return base / "autostart" / NAME


def desktop_entry(python: str) -> str:
    quoted = '"' + python.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return ("[Desktop Entry]\nType=Application\nName=ORBIT Gesture Studio\nComment=Gesture control and assistant orb\n"
            f"Exec={quoted} -m orbit hud\nTerminal=false\nX-GNOME-Autostart-enabled=true\n")


def enable(path: Optional[Path] = None, python: Optional[str] = None) -> Path:
    if not sys.platform.startswith("linux") and path is None:
        raise RuntimeError("autostart is only implemented for Linux (XDG) so far")
    path = Path(path) if path else autostart_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(desktop_entry(python or sys.executable), encoding="utf-8")
    return path


def disable(path: Optional[Path] = None) -> bool:
    path = Path(path) if path else autostart_path()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
