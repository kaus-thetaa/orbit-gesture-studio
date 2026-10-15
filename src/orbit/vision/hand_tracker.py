"""MediaPipe Tasks HandLandmarker wrapper.

Frames passed in must already be mirrored (selfie view); MediaPipe's Left/Right labels
assume a mirrored image, so mirroring first makes them correct for the user.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import ensure_model
from ..settings import RecognitionSettings
from .types import HandObservation

log = logging.getLogger(__name__)


class HandTracker:
    def __init__(self, cfg: RecognitionSettings, model_path: Optional[Path] = None) -> None:
        import mediapipe as mp  # heavy import kept lazy
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        path = Path(model_path) if model_path else ensure_model("hand_landmarker")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=cfg.min_hand_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[HandObservation]:
        import cv2

        ts = int(timestamp_ms)
        if ts <= self._last_ts:  # MediaPipe requires strictly increasing timestamps
            ts = self._last_ts + 1
        self._last_ts = ts
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        pts = result.hand_landmarks[0]
        lm = np.array([[p.x * w, p.y * h, p.z * w] for p in pts], dtype=np.float64)
        cat = result.handedness[0][0] if result.handedness and result.handedness[0] else None
        return HandObservation(
            landmarks=lm,
            handedness=(cat.category_name if cat else "Right"),
            handedness_score=float(cat.score) if cat else 0.0,
            frame_size=(w, h),
        )

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover
            pass
