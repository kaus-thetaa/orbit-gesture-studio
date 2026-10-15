"""Plain data types shared by the vision modules (no heavy dependencies)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class HandObservation:
    """One detected hand.

    landmarks: (21, 3) array in *pixel* units (x, y in pixels, z scaled like x) so that
    distances are not distorted by the frame's aspect ratio.
    handedness: "Left"/"Right" as seen by the user (frames are mirrored before inference).
    """

    landmarks: np.ndarray
    handedness: str
    handedness_score: float
    frame_size: Tuple[int, int]  # (width, height)

    def size(self) -> float:
        """Wrist -> middle-finger knuckle distance in pixels (a stable hand-size measure)."""
        return float(np.linalg.norm(self.landmarks[9, :2] - self.landmarks[0, :2]))


@dataclass(frozen=True)
class GesturePrediction:
    name: str
    confidence: float
    source: str  # "builtin" | "custom"


@dataclass(frozen=True)
class GestureEvent:
    kind: str  # "pressed" | "released"
    name: str
    source: str  # "hand" | "gaze" | "face"
    timestamp: float


@dataclass(frozen=True)
class FaceObservation:
    gaze_ratio: Optional[float]  # 0 = looking to the image-left, 1 = image-right (mirrored view)
    blendshapes: dict
