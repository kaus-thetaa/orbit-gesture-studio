"""Application settings: typed, validated, forgiving.

Bad values in settings.json never crash the app - they are clamped or replaced with
defaults and a warning is logged. (Mappings, which can trigger actions, are validated
strictly elsewhere.)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

from . import paths
from .fsutil import atomic_write_json, quarantine_corrupt

log = logging.getLogger(__name__)


def _f(default, lo=None, hi=None, doc=""):
    md = {"doc": doc}
    if lo is not None:
        md["min"] = lo
    if hi is not None:
        md["max"] = hi
    return field(default=default, metadata=md)


@dataclass
class CameraSettings:
    index: int = _f(0, 0, 10, "Camera index (0 = default webcam)")
    width: int = _f(640, 160, 1920, "Requested capture width")
    height: int = _f(480, 120, 1080, "Requested capture height")
    fps: int = _f(30, 5, 60, "Requested frames per second")
    backend: str = _f("auto", doc="auto | dshow | msmf | any")
    fourcc: str = _f("MJPG", doc="Pixel format hint; empty string to skip")
    reconnect_interval_s: float = _f(2.0, 0.5, 30.0, "Seconds between reconnect attempts")
    read_failure_limit: int = _f(30, 3, 600, "Failed reads before declaring the camera lost")
    stale_after_s: float = _f(1.0, 0.2, 10.0, "Frame age after which the feed counts as lost")


@dataclass
class RecognitionSettings:
    min_hand_detection_confidence: float = _f(0.6, 0.1, 0.99, "MediaPipe hand detection threshold")
    min_hand_presence_confidence: float = _f(0.6, 0.1, 0.99, "MediaPipe hand presence threshold")
    min_tracking_confidence: float = _f(0.5, 0.1, 0.99, "MediaPipe tracking threshold")
    min_gesture_confidence: float = _f(0.65, 0.2, 0.99, "Minimum confidence for a frame to count")
    confirm_frames: int = _f(6, 1, 60, "Consecutive frames needed to activate a gesture")
    release_frames: int = _f(6, 1, 120, "Consecutive mismatching frames needed to release it")
    extension_ratio: float = _f(1.35, 1.05, 2.5, "tip-to-wrist / knuckle-to-wrist for an extended finger")
    straight_angle_deg: float = _f(140.0, 100.0, 175.0, "Minimum PIP joint angle for an extended finger")
    thumb_ratio: float = _f(0.8, 0.3, 1.6, "Thumb-tip to index-knuckle distance / palm width for thumb 'out'")
    min_hand_size: float = _f(0.05, 0.01, 0.5, "Wrist-to-knuckle length as fraction of frame diagonal")
    edge_margin: float = _f(0.01, 0.0, 0.2, "Tolerance for landmarks outside the frame before rejecting")
    custom_distance_scale: float = _f(2.5, 1.0, 10.0, "Custom gesture threshold = scale x typical sample spread")


@dataclass
class SafetySettings:
    start_armed: bool = _f(False, doc="Start with actions enabled (NOT recommended)")
    emergency_hotkey: str = _f("<ctrl>+<alt>+<shift>+e", doc="Global emergency stop")
    arm_hotkey: str = _f("<ctrl>+<alt>+<shift>+g", doc="Global arm/disarm toggle")
    enable_hotkeys: bool = _f(True, doc="Register global hotkeys")
    shortcut_denylist: list = field(
        default_factory=lambda: ["alt+f4", "ctrl+alt+delete", "win+l", "ctrl+shift+esc", "win+r", "ctrl+w"]
    )
    platform_shortcuts: dict = field(default_factory=dict)


@dataclass
class GazeSettings:
    enabled: bool = _f(False, doc="EXPERIMENTAL gaze signals (needs face model)")
    left_threshold: float = _f(0.40, 0.05, 0.5, "Iris ratio at or below this = looking left")
    right_threshold: float = _f(0.60, 0.5, 0.95, "Iris ratio at or above this = looking right")
    smoothing_frames: int = _f(5, 1, 30, "Moving-average window for the iris ratio")
    dwell_ms: int = _f(900, 200, 5000, "Look duration needed before the signal fires")
    release_ms: int = _f(300, 50, 3000, "Time away from the region before the signal releases")
    cooldown_ms: int = _f(1500, 0, 10000, "Minimum gap between firings")


@dataclass
class FaceSettings:
    enabled: bool = _f(False, doc="EXPERIMENTAL face-blendshape signals (needs face model)")
    # Each signal: {"name": "jaw_open", "blendshape": "jawOpen", "threshold": 0.6, "dwell_ms": 400}
    signals: list = field(default_factory=list)


@dataclass
class UISettings:
    show_landmarks: bool = _f(True)
    preview_fps: int = _f(20, 5, 60)


@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    gaze: GazeSettings = field(default_factory=GazeSettings)
    face: FaceSettings = field(default_factory=FaceSettings)
    ui: UISettings = field(default_factory=UISettings)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        s = cls()
        if not isinstance(data, dict):
            log.warning("settings: expected an object, using defaults")
            return s
        for f in fields(cls):
            section = getattr(s, f.name)
            raw = data.get(f.name)
            if raw is None:
                continue
            if not isinstance(raw, dict):
                log.warning("settings: section %r is not an object, ignored", f.name)
                continue
            _fill_section(section, raw, f.name)
        return s

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        path = Path(path) if path else paths.settings_path()
        if not path.exists():
            s = cls()
            try:
                s.save(path)
            except OSError as exc:  # read-only FS etc. - not fatal
                log.warning("could not write default settings: %s", exc)
            return s
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("settings unreadable (%s); using defaults", exc)
            try:
                quarantine_corrupt(path)
            except OSError:
                pass
            return cls()
        return cls.from_dict(data)

    def save(self, path: Optional[Path] = None) -> None:
        atomic_write_json(Path(path) if path else paths.settings_path(), self.to_dict())


def _coerce(value: Any, default: Any) -> Any:
    t = type(default)
    if t is bool:
        if isinstance(value, bool):
            return value
        raise ValueError("expected true/false")
    if t is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a number")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("expected a whole number")
        return int(value)
    if t is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a number")
        return float(value)
    if t is str:
        if not isinstance(value, str):
            raise ValueError("expected text")
        return value
    if t is list:
        if not isinstance(value, list):
            raise ValueError("expected a list")
        return value
    if t is dict:
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    return value


def _fill_section(section: Any, raw: dict, name: str) -> None:
    for f in fields(section):
        if f.name not in raw:
            continue
        current = getattr(section, f.name)
        try:
            value = _coerce(raw[f.name], current)
        except ValueError as exc:
            log.warning("settings: %s.%s invalid (%s); keeping default %r", name, f.name, exc, current)
            continue
        lo, hi = f.metadata.get("min"), f.metadata.get("max")
        if lo is not None and hi is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            if not (lo <= value <= hi):
                clamped = type(current)(min(max(value, lo), hi))
                log.warning("settings: %s.%s=%r out of range [%s, %s]; clamped to %r", name, f.name, value, lo, hi, clamped)
                value = clamped
        setattr(section, f.name, value)
