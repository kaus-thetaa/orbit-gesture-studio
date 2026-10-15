"""`orbit doctor`: environment report to paste into bug reports / troubleshooting."""
from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from typing import List

from . import models, paths

PACKAGES = ["mediapipe", "opencv-contrib-python", "opencv-python", "opencv-python-headless", "numpy", "PySide6", "pynput"]


def linux_session_info(env=None, listdir=None, groups=None) -> List[str]:
    """Display-server, camera-device and permission facts that decide whether ORBIT works on Linux."""
    import os
    env = os.environ if env is None else env
    listdir = listdir or (lambda p: sorted(os.listdir(p)))
    out = ["Linux session:"]
    stype = env.get("XDG_SESSION_TYPE", "unknown")
    out.append(f"  XDG_SESSION_TYPE   {stype}   DISPLAY={env.get('DISPLAY', '-')}   WAYLAND_DISPLAY={env.get('WAYLAND_DISPLAY', '-')}")
    out.append(f"  Desktop            {env.get('XDG_CURRENT_DESKTOP', 'unknown')}")
    if stype == "wayland":
        out.append("  PROBLEM: Wayland blocks synthetic key events and global hotkeys, so actions/snap/hotkeys will NOT work.")
        out.append("           Log out and choose an X11 session (e.g. 'GNOME on Xorg') at the login screen.")
    try:
        cams = [d for d in listdir("/dev") if d.startswith("video")]
    except OSError:
        cams = []
    out.append(f"  /dev/video*        {', '.join(cams) if cams else 'NONE FOUND (is the webcam connected / driver loaded?)'}")
    try:
        if groups is None:
            import grp
            user = env.get("USER") or os.getlogin()
            groups = [g.gr_name for g in grp.getgrall() if user in g.gr_mem] + [grp.getgrgid(os.getgid()).gr_name]
        out.append(f"  In 'video' group   {'yes' if 'video' in groups else 'NO - run: sudo usermod -aG video $USER  (then log out/in)'}")
    except Exception:
        out.append("  In 'video' group   (could not determine)")
    return out


def collect() -> List[str]:
    lines = [
        f"Python      : {sys.version.split()[0]} ({platform.python_implementation()}, {platform.architecture()[0]})",
        f"Platform    : {platform.platform()}",
        f"Config dir  : {paths.config_dir()}",
        "Packages    :",
    ]
    for pkg in PACKAGES:
        try:
            lines.append(f"  {pkg:<28} {md.version(pkg)}")
        except md.PackageNotFoundError:
            lines.append(f"  {pkg:<28} (not installed)")
    both = [p for p in ("opencv-python", "opencv-contrib-python") if _has(p)]
    if len(both) > 1:
        lines.append("  WARNING: opencv-python AND opencv-contrib-python are both installed; this can break `import cv2`.")
        lines.append("           Fix: pip uninstall -y opencv-python opencv-python-headless && pip install --force-reinstall opencv-contrib-python")
    lines.append("Models      :")
    for name in models.MODELS:
        p = models.model_path(name)
        lines.append(f"  {name:<18} {'present' if p.exists() else 'MISSING (run: orbit download-models)'}  [{p}]")
    try:
        import cv2
        lines.append(f"cv2         : {cv2.__version__}")
    except Exception as exc:
        lines.append(f"cv2         : IMPORT FAILED: {exc}")
    try:
        from PySide6 import __version__ as pv
        lines.append(f"PySide6     : {pv}")
    except Exception as exc:
        lines.append(f"PySide6     : IMPORT FAILED: {exc}")
    if sys.platform.startswith("linux"):
        lines += linux_session_info()
    return lines


def _has(pkg: str) -> bool:
    try:
        md.version(pkg)
        return True
    except md.PackageNotFoundError:
        return False
