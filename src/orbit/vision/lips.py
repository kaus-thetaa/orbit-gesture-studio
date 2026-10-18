"""Finger-on-lips ("shh") detection: a hand-pose gesture that needs to know where the mouth is.

Cheap by design: the face model runs ONLY when a one-finger pose is already being seen, at most
every `check_interval_ms`, and the mouth position is cached for `face_cache_ms`.

Geometry: mouth centre = mean of 4 lip landmarks; distance unit = mouth width, so it doesn't
depend on how far you sit. A depth sanity check (hand size vs mouth width) rejects a far-away
hand that merely overlaps the mouth in the image.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Tuple

import numpy as np

from ..settings import LipsSettings
from .types import HandObservation

log = logging.getLogger(__name__)

LIP_IDS = (13, 14, 61, 291)       # inner upper, inner lower, left corner, right corner (Face Landmarker 478-pt)
INDEX_TIP = 8


def mouth_geometry(xy: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    if xy is None or xy.shape[0] < 292:
        return None
    width = float(np.linalg.norm(xy[61] - xy[291]))
    if width < 4.0:                                  # face too small/degenerate in the image
        return None
    return xy[list(LIP_IDS)].mean(axis=0), width


def lips_confidence(tip: np.ndarray, center: np.ndarray, width: float, max_distance: float) -> float:
    d = float(np.linalg.norm(tip - center)) / width
    return max(0.0, min(1.0, 2.0 * (1.0 - d / max_distance)))


class MouthTracker:
    def __init__(self, cfg: LipsSettings, face_fn: Callable[[np.ndarray, int], Optional[np.ndarray]]) -> None:
        self.cfg, self.face_fn = cfg, face_fn
        self._geom: Optional[Tuple[np.ndarray, float]] = None
        self._geom_t = -1e9
        self._last_try = -1e9

    def check(self, frame: np.ndarray, hand: HandObservation, ts_ms: int) -> Optional[float]:
        """Confidence (0-1) that the index fingertip is on the lips, or None if it can't be told."""
        now = ts_ms / 1000.0
        cfg = self.cfg
        fresh = self._geom is not None and now - self._geom_t <= cfg.face_cache_ms / 1000.0
        if not fresh and now - self._last_try >= cfg.check_interval_ms / 1000.0:
            self._last_try = now
            try:
                self._geom = mouth_geometry(self.face_fn(frame, ts_ms))
            except Exception:
                log.exception("face landmarks failed")
                self._geom = None
            self._geom_t = now
        if self._geom is None or now - self._geom_t > cfg.face_cache_ms / 1000.0:
            return None
        center, width = self._geom
        ratio = hand.size() / width
        if not 0.8 <= ratio <= 5.0:                  # hand much nearer/farther than the face: not "on the lips"
            return None
        return lips_confidence(hand.landmarks[INDEX_TIP, :2], center, width, cfg.max_distance)
