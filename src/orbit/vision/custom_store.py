"""Persistence + recording for user-defined static gestures.

Only normalised landmark numbers are stored - never camera frames - so custom gesture
files contain no images of the user.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .. import paths
from ..errors import ConfigError
from ..fsutil import atomic_write_json, quarantine_corrupt
from ..settings import RecognitionSettings
from .gestures import CustomGesture, CustomGestureClassifier, builtin_matches
from .quality import check_hand_quality
from .types import HandObservation

log = logging.getLogger(__name__)


class CustomGestureStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else paths.custom_gestures_path()
        self.gestures: List[CustomGesture] = []

    def load(self) -> List[CustomGesture]:
        self.gestures = []
        if not self.path.exists():
            return self.gestures
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for raw in data.get("gestures", []):
                g = CustomGesture(
                    name=raw["name"],
                    samples=np.asarray(raw["samples"], dtype=np.float64),
                    sensitivity=float(raw.get("sensitivity", 1.0)),
                    created=str(raw.get("created", "")),
                )
                g.validate()
                self.gestures.append(g)
        except (OSError, ValueError, KeyError, TypeError, ConfigError) as exc:
            log.error("custom gestures unreadable (%s)", exc)
            try:
                quarantine_corrupt(self.path)
            except OSError:
                pass
            self.gestures = []
        return self.gestures

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "gestures": [
                {
                    "name": g.name,
                    "sensitivity": g.sensitivity,
                    "created": g.created,
                    "samples": np.round(g.samples, 5).tolist(),
                }
                for g in self.gestures
            ],
        })

    def get(self, name: str) -> Optional[CustomGesture]:
        return next((g for g in self.gestures if g.name == name), None)

    def add(self, gesture: CustomGesture, replace: bool = False) -> None:
        gesture.validate()
        existing = self.get(gesture.name)
        if existing and not replace:
            raise ConfigError(f"A custom gesture named {gesture.name!r} already exists")
        self.gestures = [g for g in self.gestures if g.name != gesture.name] + [gesture]
        self.save()

    def remove(self, name: str) -> bool:
        before = len(self.gestures)
        self.gestures = [g for g in self.gestures if g.name != name]
        if len(self.gestures) != before:
            self.save()
            return True
        return False


class GestureRecorder:
    """Collects samples for one new custom gesture."""

    def __init__(self, cfg: RecognitionSettings) -> None:
        self.cfg = cfg
        self.samples: List[np.ndarray] = []

    def try_capture(self, hand: Optional[HandObservation]) -> str:
        """Add a sample. Returns "" on success, otherwise the reason it was rejected."""
        if hand is None:
            return "no hand detected"
        reason = check_hand_quality(hand, self.cfg)
        if reason:
            return {"partial_hand": "hand is partly outside the frame", "hand_too_small": "hand is too far away"}.get(reason, reason)
        if len(self.samples) >= CustomGesture.MAX_SAMPLES:
            return "sample limit reached"
        self.samples.append(CustomGestureClassifier.features(hand))
        return ""

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.samples):
            del self.samples[index]

    def clear(self) -> None:
        self.samples.clear()

    def as_array(self) -> np.ndarray:
        return np.vstack(self.samples) if self.samples else np.zeros((0, 42))

    def outlier_indices(self) -> List[int]:
        """Samples far from all the others - likely a bad capture (motion blur, wrong pose)."""
        n = len(self.samples)
        if n < 5:
            return []
        arr = self.as_array()
        d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
        score = np.median(d, axis=1)
        med = float(np.median(score))
        mad = float(np.median(np.abs(score - med))) or 1e-6
        return [i for i, s in enumerate(score) if s > med + 4.0 * mad and s > 1.5 * med]

    def builtin_overlap(self) -> dict:
        return {k: v for k, v in builtin_matches(self.as_array(), self.cfg).items() if v >= 0.5}

    def build(self, name: str, sensitivity: float = 1.0) -> CustomGesture:
        g = CustomGesture(
            name=name.strip(),
            samples=self.as_array(),
            sensitivity=sensitivity,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        g.validate()
        return g
