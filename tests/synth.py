"""Synthetic hand-landmark generator for tests (no camera / MediaPipe needed).

Builds a plausible 21-point hand in a clean 2D frame, with each finger interpolated
between a straight pose (t=1) and a curled pose (t=0), then rotates/scales/mirrors/places
it in pixel space. This validates the *geometry code*; it does NOT prove the thresholds
are right for real webcam data - see docs/CALIBRATION.md.
"""
from __future__ import annotations

import math
from typing import Dict, Union

import numpy as np

from orbit.vision.types import HandObservation

FRAME = (640, 480)

_MCP = {"index": (-2.5, 9.0), "middle": (-0.8, 9.4), "ring": (0.9, 9.0), "pinky": (2.5, 8.0)}
_LEN = {"index": (3.5, 2.2, 1.8), "middle": (3.9, 2.5, 2.0), "ring": (3.6, 2.3, 1.9), "pinky": (2.8, 1.6, 1.6)}

_THUMB_OUT = [(-2.0, 1.5), (-3.6, 3.2), (-5.0, 4.8), (-6.4, 6.2)]
_THUMB_TUCK = [(-2.0, 1.5), (-2.6, 2.6), (-1.8, 4.6), (-1.0, 6.0)]


def _finger_points(name: str, t: float):
    mx, my = _MCP[name]
    l1, l2, l3 = _LEN[name]
    ext = [(mx, my), (mx, my + l1), (mx, my + l1 + l2), (mx, my + l1 + l2 + l3)]
    cur = [(mx, my), (mx, my + 1.2), (mx, my - 1.0), (mx, my - 2.6)]
    return [((1 - t) * c[0] + t * e[0], (1 - t) * c[1] + t * e[1]) for c, e in zip(cur, ext)]


def make_hand(
    fingers: Dict[str, Union[bool, float]] | None = None,
    thumb: Union[bool, float] = False,
    rotation_deg: float = 0.0,
    scale: float = 12.0,
    center=(320.0, 330.0),
    handedness: str = "Right",
    noise: float = 0.0,
    seed: int = 0,
    frame_size=FRAME,
) -> HandObservation:
    fingers = fingers or {}
    pts = [(0.0, 0.0)]
    tt = float(thumb)
    pts += [tuple((1 - tt) * np.array(c) + tt * np.array(o)) for c, o in zip(_THUMB_TUCK, _THUMB_OUT)]
    for name in ("index", "middle", "ring", "pinky"):
        pts += _finger_points(name, float(fingers.get(name, False)))
    arr = np.array(pts, dtype=float)
    if handedness == "Left":
        arr[:, 0] = -arr[:, 0]
    th = math.radians(rotation_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    arr = arr @ rot.T
    if noise:
        arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
    px = np.zeros((21, 3))
    px[:, 0] = center[0] + scale * arr[:, 0]
    px[:, 1] = center[1] - scale * arr[:, 1]
    return HandObservation(px, handedness, 0.95, frame_size)


OPEN = dict(index=True, middle=True, ring=True, pinky=True)
TWO = dict(index=True, middle=True)
FIST = dict()
