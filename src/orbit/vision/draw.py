"""OpenCV overlay drawing for the live preview."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import HandObservation

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def draw_hand(frame: np.ndarray, hand: HandObservation, ok: bool = True) -> None:
    color = (80, 220, 120) if ok else (60, 60, 230)  # BGR
    pts = hand.landmarks[:, :2].astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_label(frame: np.ndarray, text: str, org=(10, 28), color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
