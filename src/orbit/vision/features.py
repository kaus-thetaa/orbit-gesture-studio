"""Geometric hand features computed from landmarks (no raw pixels).

Landmark indices follow MediaPipe Hands: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
13-16 ring, 17-20 pinky (MCP, PIP, DIP, TIP for the four fingers).

Finger extension uses two orientation-independent cues:
  * ratio  = |tip - wrist| / |mcp - wrist|   (an extended finger reaches far past its knuckle)
  * angle  = angle at the PIP joint between MCP and DIP (a straight finger is ~180 deg)
Each is turned into a signed margin in [-1, 1]; the finger's margin is the smaller of the
two, so a finger only counts as extended when BOTH cues agree, and |margin| doubles as a
confidence for that decision. Thumbs use the thumb-tip distance from the index knuckle,
normalised by palm width.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..settings import RecognitionSettings

WRIST = 0
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

_RATIO_SPAN = 0.5   # ratio distance from threshold that counts as "fully sure"
_ANGLE_SPAN = 30.0  # degrees from threshold that counts as "fully sure"
_THUMB_SPAN = 0.4


@dataclass(frozen=True)
class FingerState:
    extended: bool
    margin: float  # signed, in [-1, 1]; > 0 extended, < 0 curled

    @property
    def confidence(self) -> float:
        return min(1.0, abs(self.margin) / 0.3)


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b formed by a-b-c, degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def finger_state(lm: np.ndarray, finger: str, cfg: RecognitionSettings) -> FingerState:
    mcp, pip, dip, tip = (lm[i, :2] for i in FINGERS[finger])
    wrist = lm[WRIST, :2]
    base = float(np.linalg.norm(mcp - wrist))
    ratio = float(np.linalg.norm(tip - wrist)) / base if base > 1e-9 else 0.0
    angle = _angle_deg(mcp, pip, dip)
    rm = _clip((ratio - cfg.extension_ratio) / _RATIO_SPAN)
    am = _clip((angle - cfg.straight_angle_deg) / _ANGLE_SPAN)
    margin = min(rm, am)
    return FingerState(extended=margin > 0, margin=margin)


def thumb_state(lm: np.ndarray, cfg: RecognitionSettings) -> FingerState:
    tip = lm[4, :2]
    index_mcp, pinky_mcp = lm[5, :2], lm[17, :2]
    palm_w = float(np.linalg.norm(index_mcp - pinky_mcp))
    if palm_w < 1e-9:
        return FingerState(False, 0.0)
    r = float(np.linalg.norm(tip - index_mcp)) / palm_w
    margin = _clip((r - cfg.thumb_ratio) / _THUMB_SPAN)
    return FingerState(margin > 0, margin)


def finger_states(lm: np.ndarray, cfg: RecognitionSettings) -> Dict[str, FingerState]:
    out = {"thumb": thumb_state(lm, cfg)}
    for name in FINGERS:
        out[name] = finger_state(lm, name, cfg)
    return out


def normalize_landmarks(lm: np.ndarray, handedness: str) -> np.ndarray:
    """Translation-, scale-, rotation- and handedness-invariant (21, 2) landmarks.

    The wrist is moved to the origin, the wrist->middle-knuckle vector is scaled to length
    1 and rotated to point along +y, and left hands are mirrored so that a pose recorded
    with one hand still matches the other.
    """
    pts = np.asarray(lm[:, :2], dtype=np.float64) - lm[WRIST, :2]
    axis = pts[9]
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return np.zeros((21, 2))
    pts = pts / length
    vx, vy = axis / length
    phi = np.arctan2(vx, vy)
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    pts = pts @ rot.T
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]
    return pts
