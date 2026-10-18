"""Gesture recognisers and the registry that combines them.

* FingerPatternRecognizer - built-in static poses defined as finger-extension patterns.
* CustomGestureClassifier - user-recorded static poses, nearest-neighbour on normalised
  landmarks (stored as numbers only - no images).
* GestureRegistry         - runs every registered recogniser and returns the best match
  above the confidence threshold. Custom gestures win over built-ins when both match.

Static poses only: a gesture that is defined by *motion* (a swipe, a wave) needs a
temporal model (e.g. DTW or a small sequence network over landmark trajectories) and is
not supported here.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from ..errors import ConfigError
from ..settings import RecognitionSettings
from .features import finger_states, normalize_landmarks
from .motion import BUILTIN_MOTIONS
from .types import GesturePrediction, HandObservation

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
RESERVED_PREFIXES = ("gaze_", "face_")
CONTEXTUAL_GESTURES = ("finger_on_lips",)

# pattern values: True = must be extended, False = must be curled, None = don't care
BUILTIN_PATTERNS: Dict[str, Dict[str, Optional[bool]]] = {
    "open_palm":     {"thumb": None, "index": True,  "middle": True,  "ring": True,  "pinky": True},
    "fist":          {"thumb": None, "index": False, "middle": False, "ring": False, "pinky": False},
    "one_finger":    {"thumb": None, "index": True,  "middle": False, "ring": False, "pinky": False},
    "two_fingers":   {"thumb": None, "index": True,  "middle": True,  "ring": False, "pinky": False},
    "three_fingers": {"thumb": None, "index": True,  "middle": True,  "ring": True,  "pinky": False},
}


class Recognizer(Protocol):
    priority: int

    def classify(self, hand: HandObservation) -> List[GesturePrediction]: ...


class FingerPatternRecognizer:
    priority = 0

    def __init__(self, cfg: RecognitionSettings, patterns: Optional[Dict[str, dict]] = None) -> None:
        self.cfg = cfg
        self.patterns = dict(patterns or BUILTIN_PATTERNS)

    def names(self) -> List[str]:
        return list(self.patterns)

    def classify(self, hand: HandObservation) -> List[GesturePrediction]:
        states = finger_states(hand.landmarks, self.cfg)
        out: List[GesturePrediction] = []
        for name, pattern in self.patterns.items():
            conf = 1.0
            ok = True
            for finger, want in pattern.items():
                if want is None:
                    continue
                st = states[finger]
                if st.extended != want:
                    ok = False
                    break
                conf = min(conf, st.confidence)
            if ok:
                out.append(GesturePrediction(name, conf, "builtin"))
        return out


# ---------------------------------------------------------------- custom gestures
@dataclass
class CustomGesture:
    name: str
    samples: np.ndarray  # (N, 42)
    sensitivity: float = 1.0
    created: str = ""
    threshold: float = field(default=0.0)

    MIN_SAMPLES = 8
    MAX_SAMPLES = 200

    def validate(self) -> None:
        if not NAME_RE.match(self.name):
            raise ConfigError("Gesture name must be 2-32 chars: lowercase letters, digits, underscore; start with a letter")
        if self.name in BUILTIN_PATTERNS or self.name in BUILTIN_MOTIONS or self.name in CONTEXTUAL_GESTURES or self.name.startswith(RESERVED_PREFIXES):
            raise ConfigError(f"{self.name!r} is reserved")
        if self.samples.ndim != 2 or self.samples.shape[1] != 42:
            raise ConfigError("Invalid sample shape")
        if not (self.MIN_SAMPLES <= len(self.samples) <= self.MAX_SAMPLES):
            raise ConfigError(f"Need between {self.MIN_SAMPLES} and {self.MAX_SAMPLES} samples (have {len(self.samples)})")
        if not np.isfinite(self.samples).all():
            raise ConfigError("Samples contain invalid numbers")
        if not (0.2 <= self.sensitivity <= 5.0):
            raise ConfigError("sensitivity must be between 0.2 and 5.0")

    def spread(self) -> float:
        """Median leave-one-out nearest-neighbour distance between samples."""
        d = np.linalg.norm(self.samples[:, None, :] - self.samples[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(np.median(d.min(axis=1)))


class CustomGestureClassifier:
    """k-NN over normalised landmark vectors with a per-gesture distance threshold."""

    priority = 10
    K = 3
    FLOOR = 0.35
    CAP = 1.5

    def __init__(self, cfg: RecognitionSettings) -> None:
        self.cfg = cfg
        self._gestures: Dict[str, CustomGesture] = {}
        self._lock = threading.Lock()

    def set_gestures(self, gestures: Sequence[CustomGesture]) -> None:
        with self._lock:
            self._gestures = {g.name: g for g in gestures}

    def names(self) -> List[str]:
        with self._lock:
            return list(self._gestures)

    def threshold_for(self, g: CustomGesture) -> float:
        base = max(self.FLOOR, self.cfg.custom_distance_scale * g.spread())
        return min(self.CAP, base) * g.sensitivity

    @staticmethod
    def features(hand: HandObservation) -> np.ndarray:
        return normalize_landmarks(hand.landmarks, hand.handedness).reshape(-1)

    def distance(self, g: CustomGesture, feat: np.ndarray) -> float:
        d = np.sort(np.linalg.norm(g.samples - feat, axis=1))
        return float(d[: min(self.K, len(d))].mean())

    def confidence(self, g: CustomGesture, feat: np.ndarray) -> float:
        thr = self.threshold_for(g)
        d = self.distance(g, feat)
        return float(max(0.0, min(1.0, 2.0 * (1.0 - d / thr))))

    def classify(self, hand: HandObservation) -> List[GesturePrediction]:
        feat = self.features(hand)
        with self._lock:
            gestures = list(self._gestures.values())
        return [GesturePrediction(g.name, c, "custom") for g in gestures if (c := self.confidence(g, feat)) > 0.0]


# ---------------------------------------------------------------- registry
class GestureRegistry:
    def __init__(self, cfg: RecognitionSettings) -> None:
        self.cfg = cfg
        self.recognizers: List[Recognizer] = []

    def register(self, rec: Recognizer) -> None:
        self.recognizers.append(rec)

    def classify(self, hand: HandObservation) -> Optional[GesturePrediction]:
        best: Optional[GesturePrediction] = None
        best_key = (-1, -1.0)
        for rec in self.recognizers:
            for pred in rec.classify(hand):
                if pred.confidence < self.cfg.min_gesture_confidence:
                    continue
                key = (rec.priority, pred.confidence)
                if key > best_key:
                    best, best_key = pred, key
        return best

    def hand_gesture_names(self) -> List[str]:
        names: List[str] = []
        for rec in self.recognizers:
            if hasattr(rec, "names"):
                names.extend(rec.names())  # type: ignore[attr-defined]
        return names


def builtin_matches(samples: np.ndarray, cfg: RecognitionSettings) -> Dict[str, float]:
    """For a recorded sample set, the fraction of samples that a built-in pose also matches.

    Used by the recorder to warn when a 'custom' gesture is really just 'fist' etc.
    Works on normalised samples because the extension cues are scale/rotation invariant.
    """
    rec = FingerPatternRecognizer(cfg)
    counts: Dict[str, int] = {}
    for row in samples:
        lm = np.zeros((21, 3))
        lm[:, :2] = row.reshape(21, 2)
        hand = HandObservation(lm, "Right", 1.0, (1, 1))
        for p in rec.classify(hand):
            if p.confidence >= cfg.min_gesture_confidence:
                counts[p.name] = counts.get(p.name, 0) + 1
    return {k: v / max(1, len(samples)) for k, v in counts.items()}
