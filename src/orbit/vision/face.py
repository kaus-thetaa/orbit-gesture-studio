"""EXPERIMENTAL: webcam gaze approximation and face-blendshape signals.

Everything here is optional, off by default, and isolated from the core hand pipeline.

Gaze: the horizontal position of each iris between the two eye corners (0 = image-left,
1 = image-right). This is a *relative eye-in-head* estimate, not a screen coordinate:
turning your head changes it, glasses/lighting/camera angle matter, and it needs a
calibration (see `orbit calibrate-gaze`). A dwell time and cooldown prevent ordinary
reading/scanning from firing actions.

Face signals: user-defined thresholds on MediaPipe face blendshape scores (e.g. jawOpen).
These are raw muscle-movement scores - NOT emotion detection - and are just another
trigger source alongside gestures.
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..errors import ConfigError
from ..models import ensure_model
from ..settings import FaceSettings, GazeSettings
from .stabilizer import DwellDetector
from .types import FaceObservation, GestureEvent

log = logging.getLogger(__name__)

# MediaPipe Face Landmarker (478-point) indices. Corners are ordered image-left -> image-right.
_EYE_A = (33, 133, 468)    # corners (image-left, image-right), iris centre
_EYE_B = (362, 263, 473)
SIGNAL_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


def iris_ratio(xy: np.ndarray) -> Optional[float]:
    """Mean horizontal iris position across both eyes (0..1), or None if geometry is degenerate."""
    if xy.shape[0] < 478:
        return None
    ratios = []
    for left, right, iris in (_EYE_A, _EYE_B):
        width = xy[right, 0] - xy[left, 0]
        if width < 1e-4:
            return None
        ratios.append((xy[iris, 0] - xy[left, 0]) / width)
    return float(np.mean(ratios))


class FaceModule:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        path = Path(model_path) if model_path else ensure_model("face_landmarker")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[FaceObservation]:
        import cv2

        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.face_landmarks:
            return None
        xy = np.array([[p.x, p.y] for p in res.face_landmarks[0]])
        shapes = {}
        if res.face_blendshapes:
            shapes = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
        return FaceObservation(gaze_ratio=iris_ratio(xy), blendshapes=shapes)

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:  # pragma: no cover
            pass


class FaceSignalHub:
    """Turns face observations into dwell-filtered pressed/released events."""

    def __init__(self, gaze: GazeSettings, face: FaceSettings, clock: Callable[[], float] = time.monotonic) -> None:
        self.gaze_cfg, self.face_cfg, self.clock = gaze, face, clock
        self._ratios: Deque[float] = deque(maxlen=max(1, gaze.smoothing_frames))
        self._gaze: Dict[str, DwellDetector] = {}
        self._face: List[Tuple[dict, DwellDetector]] = []
        self.rebuild()

    def rebuild(self) -> None:
        g = self.gaze_cfg
        self._ratios = deque(maxlen=max(1, g.smoothing_frames))
        self._gaze = {
            name: DwellDetector(name, "gaze", g.dwell_ms, g.release_ms, g.cooldown_ms, self.clock)
            for name in ("gaze_left", "gaze_right")
        }
        self._face = []
        for sig in self.face_cfg.signals:
            try:
                spec = validate_signal(sig)
            except ConfigError as exc:
                log.warning("face signal ignored: %s", exc)
                continue
            det = DwellDetector(f"face_{spec['name']}", "face", spec["dwell_ms"], 200, 800, self.clock)
            self._face.append((spec, det))

    @property
    def wants_face(self) -> bool:
        return (self.gaze_cfg.enabled) or (self.face_cfg.enabled and bool(self._face))

    def update(self, obs: Optional[FaceObservation]) -> Tuple[List[GestureEvent], dict]:
        events: List[GestureEvent] = []
        debug: dict = {"gaze_ratio": None}
        if self.gaze_cfg.enabled:
            ratio = obs.gaze_ratio if obs else None
            if ratio is not None:
                self._ratios.append(ratio)
            else:
                self._ratios.clear()
            smooth = float(np.mean(self._ratios)) if self._ratios else None
            debug["gaze_ratio"] = smooth
            left = smooth is not None and smooth <= self.gaze_cfg.left_threshold
            right = smooth is not None and smooth >= self.gaze_cfg.right_threshold
            events += self._gaze["gaze_left"].update(left)
            events += self._gaze["gaze_right"].update(right)
        if self.face_cfg.enabled:
            for spec, det in self._face:
                score = obs.blendshapes.get(spec["blendshape"], 0.0) if obs else 0.0
                events += det.update(score >= spec["threshold"])
        return events, debug

    def reset(self) -> List[GestureEvent]:
        out: List[GestureEvent] = []
        for det in list(self._gaze.values()) + [d for _, d in self._face]:
            out += det.reset()
        self._ratios.clear()
        return out

    def signal_names(self) -> List[str]:
        names = []
        if self.gaze_cfg.enabled:
            names += ["gaze_left", "gaze_right"]
        if self.face_cfg.enabled:
            names += [d.name for _, d in self._face]
        return names


def validate_signal(sig: dict) -> dict:
    if not isinstance(sig, dict):
        raise ConfigError("face signal must be an object")
    name, shape = sig.get("name"), sig.get("blendshape")
    thr, dwell = sig.get("threshold", 0.6), sig.get("dwell_ms", 400)
    if not isinstance(name, str) or not SIGNAL_RE.match(name):
        raise ConfigError(f"bad face signal name {name!r}")
    if not isinstance(shape, str) or not re.match(r"^[A-Za-z]{3,40}$", shape):
        raise ConfigError(f"bad blendshape name {shape!r}")
    if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not 0.05 <= thr <= 0.99:
        raise ConfigError("threshold must be between 0.05 and 0.99")
    if isinstance(dwell, bool) or not isinstance(dwell, int) or not 100 <= dwell <= 5000:
        raise ConfigError("dwell_ms must be a whole number between 100 and 5000")
    return {"name": name, "blendshape": shape, "threshold": float(thr), "dwell_ms": dwell}


class GazeCalibrator:
    """Collects iris ratios while the user looks left / centre / right, then derives thresholds."""

    PHASES = ("left", "center", "right")

    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = {p: [] for p in self.PHASES}

    def add(self, phase: str, ratio: Optional[float]) -> None:
        if ratio is not None and phase in self.samples:
            self.samples[phase].append(float(ratio))

    def compute(self, min_samples: int = 20) -> Tuple[float, float]:
        means = {}
        for p, vals in self.samples.items():
            if len(vals) < min_samples:
                raise ConfigError(f"Not enough usable samples for '{p}' ({len(vals)}/{min_samples}); is your face visible?")
            means[p] = float(np.median(vals))
        l, c, r = means["left"], means["center"], means["right"]
        if not (c - l >= 0.04 and r - c >= 0.04):
            raise ConfigError(
                f"Eye movement too small to separate left/centre/right (left={l:.2f}, centre={c:.2f}, right={r:.2f}). "
                "Keep your head still and look clearly at each edge."
            )
        return c - 0.6 * (c - l), c + 0.6 * (r - c)
