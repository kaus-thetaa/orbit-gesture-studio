"""Reject observations that are too unreliable to classify."""
from __future__ import annotations

import math
from typing import Optional

from ..settings import RecognitionSettings
from .types import HandObservation


def check_hand_quality(hand: HandObservation, cfg: RecognitionSettings) -> Optional[str]:
    """Return None if usable, else a short machine-readable reason."""
    w, h = hand.frame_size
    xs, ys = hand.landmarks[:, 0], hand.landmarks[:, 1]
    m = cfg.edge_margin
    if xs.min() < -m * w or xs.max() > (1 + m) * w or ys.min() < -m * h or ys.max() > (1 + m) * h:
        return "partial_hand"
    if hand.size() / math.hypot(w, h) < cfg.min_hand_size:
        return "hand_too_small"
    return None
