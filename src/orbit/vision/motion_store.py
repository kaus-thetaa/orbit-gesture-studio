"""Persistence + recording for custom motion gestures (numbers only - no images)."""
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
from ..settings import MotionSettings
from .motion import (BuiltinMotionRecognizer, CustomMotion, N_POINTS, Segment, dtw_batch)

log = logging.getLogger(__name__)


class MotionStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else paths.custom_motions_path()
        self.motions: List[CustomMotion] = []

    def load(self) -> List[CustomMotion]:
        self.motions = []
        if not self.path.exists():
            return self.motions
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for raw in data.get("motions", []):
                m = CustomMotion(
                    name=raw["name"],
                    samples=np.asarray(raw["samples"], dtype=np.float64),
                    pose=str(raw.get("pose", "")),
                    sensitivity=float(raw.get("sensitivity", 1.0)),
                    created=str(raw.get("created", "")),
                )
                m.validate()
                self.motions.append(m)
        except (OSError, ValueError, KeyError, TypeError, ConfigError) as exc:
            log.error("custom motions unreadable (%s)", exc)
            try:
                quarantine_corrupt(self.path)
            except OSError:
                pass
            self.motions = []
        return self.motions

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "motions": [
                {"name": m.name, "pose": m.pose, "sensitivity": m.sensitivity, "created": m.created,
                 "samples": np.round(m.samples, 4).tolist()}
                for m in self.motions
            ],
        })

    def get(self, name: str) -> Optional[CustomMotion]:
        return next((m for m in self.motions if m.name == name), None)

    def add(self, motion: CustomMotion, replace: bool = False) -> None:
        motion.validate()
        if self.get(motion.name) and not replace:
            raise ConfigError(f"A custom motion named {motion.name!r} already exists")
        self.motions = [m for m in self.motions if m.name != motion.name] + [motion]
        self.save()

    def remove(self, name: str) -> bool:
        before = len(self.motions)
        self.motions = [m for m in self.motions if m.name != name]
        if len(self.motions) != before:
            self.save()
            return True
        return False


class MotionRecorder:
    """Collects example performances of one new motion."""

    def __init__(self, cfg: MotionSettings) -> None:
        self.cfg = cfg
        self.samples: List[np.ndarray] = []
        self.durations: List[float] = []
        self.poses: List[str] = []

    def add_segment(self, seg: Segment) -> str:
        """Returns "" on success, else why the sample was rejected."""
        if len(self.samples) >= CustomMotion.MAX_SAMPLES:
            return "sample limit reached"
        self.samples.append(seg.resampled(N_POINTS))
        self.durations.append(seg.duration)
        self.poses.append(seg.dominant_pose(self.cfg.pose_fraction))
        return ""

    def remove(self, i: int) -> None:
        if 0 <= i < len(self.samples):
            del self.samples[i], self.durations[i], self.poses[i]

    def clear(self) -> None:
        self.samples.clear(); self.durations.clear(); self.poses.clear()

    def as_array(self) -> np.ndarray:
        return np.stack(self.samples) if self.samples else np.zeros((0, N_POINTS, 2))

    def majority_pose(self, min_share: float = 0.6) -> str:
        named = [p for p in self.poses if p]
        if not named or len(named) / max(1, len(self.poses)) < min_share:
            return ""
        best = max(set(named), key=named.count)
        return best if named.count(best) / len(self.poses) >= min_share else ""

    def outlier_indices(self) -> List[int]:
        """Samples that don't resemble the others (DTW distance) - likely a botched attempt."""
        n = len(self.samples)
        if n < 6:
            return []
        arr = self.as_array()
        score = np.empty(n)
        for i in range(n):
            others = np.delete(arr, i, axis=0)
            score[i] = np.median(dtw_batch(others, arr[i]))
        med = float(np.median(score))
        mad = float(np.median(np.abs(score - med))) or 1e-6
        return [i for i, s in enumerate(score) if s > med + 4.0 * mad and s > 1.5 * med]

    def builtin_overlap(self) -> dict:
        """Fraction of samples a built-in motion rule would also accept (warn: 'this is just a swipe')."""
        rec = BuiltinMotionRecognizer(self.cfg)
        counts: dict = {}
        for s, d in zip(self.samples, self.durations):
            pred, _ = rec.classify_traj(s, d)
            if pred and pred.confidence >= self.cfg.min_confidence:
                counts[pred.name] = counts.get(pred.name, 0) + 1
        return {k: v / len(self.samples) for k, v in counts.items() if v / len(self.samples) >= 0.5}

    def build(self, name: str, require_pose: bool = True, sensitivity: float = 1.0) -> CustomMotion:
        m = CustomMotion(
            name=name.strip(), samples=self.as_array(),
            pose=self.majority_pose() if require_pose else "",
            sensitivity=sensitivity, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        m.validate()
        return m
