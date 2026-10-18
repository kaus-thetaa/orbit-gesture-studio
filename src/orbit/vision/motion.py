"""Movement (trajectory) gestures.

Pipeline
--------
1. Each frame the palm centre is turned into a TrackPoint (position, hand size, static pose).
2. MotionSegmenter cuts the continuous stream into *segments*: a segment starts when the
   hand speeds up past `start_speed` and ends after it has been still (or lost) for
   `gap_ms`. Speeds are in hand-lengths per second, so they don't depend on how far you
   sit from the camera.
3. A finished segment is turned into a trajectory in hand-length units, relative to its
   start point (so it doesn't depend on where in the frame you gesture either).
4. MotionEngine classifies it:
     * custom motions you recorded - Dynamic Time Warping (DTW) against your examples
       (plus automatically augmented variants); needs ~10-30 examples per motion
     * built-in motions (swipes, wave, circles) - geometric rules, no recording needed
   A motion is a momentary event ("trigger"); it is never "held".

Directions are as seen in the mirrored preview, i.e. from the user's own point of view:
swipe_left = hand moves toward the user's left. circle_cw = clockwise on the preview.
Image y grows downward, so "up" is negative dy.

Known weak spot: the *return stroke*. After swiping right the hand usually drifts back left,
which can look like a swipe_left. A cooldown (default 0.8 s) and the pose requirement
suppress most of these; they cannot be eliminated entirely.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import ConfigError
from ..settings import MotionSettings
from .types import GestureEvent

N_POINTS = 32
BAND = 8
EMA_ALPHA = 0.6
PALM_IDS = (0, 5, 9, 13, 17)
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

BUILTIN_MOTIONS: Tuple[str, ...] = (
    "swipe_left", "swipe_right", "swipe_up", "swipe_down", "wave", "circle_cw", "circle_ccw",
)


# ====================================================================== data
@dataclass(frozen=True)
class TrackPoint:
    t: float
    x: float
    y: float
    size: float            # hand size in pixels (wrist -> middle knuckle)
    pose: Optional[str]    # static gesture recognised this frame, if any


def palm_point(hand, t: float, pose: Optional[str]) -> TrackPoint:
    c = hand.landmarks[list(PALM_IDS), :2].mean(axis=0)
    return TrackPoint(t, float(c[0]), float(c[1]), max(hand.size(), 1e-6), pose)


@dataclass
class Segment:
    t: np.ndarray                 # (M,) seconds
    xy: np.ndarray                # (M, 2) pixels (already smoothed)
    size: np.ndarray              # (M,) hand size, pixels
    poses: List[Optional[str]]

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def ref_size(self) -> float:
        return float(np.median(self.size))

    def trajectory(self) -> np.ndarray:
        """(M, 2) positions in hand-lengths, relative to the first point."""
        return (self.xy - self.xy[0]) / self.ref_size()

    def path_length(self) -> float:
        d = np.diff(self.trajectory(), axis=0)
        return float(np.linalg.norm(d, axis=1).sum())

    def pose_fraction(self, pose: str) -> float:
        return sum(1 for p in self.poses if p == pose) / max(1, len(self.poses))

    def dominant_pose(self, min_fraction: float = 0.6) -> str:
        counts: Dict[str, int] = {}
        for p in self.poses:
            if p:
                counts[p] = counts.get(p, 0) + 1
        if not counts:
            return ""
        best = max(counts, key=counts.get)
        return best if counts[best] / max(1, len(self.poses)) >= min_fraction else ""

    def resampled(self, n: int = N_POINTS) -> np.ndarray:
        return resample(self.trajectory(), self.t - self.t[0], n)


def resample(traj: np.ndarray, t: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    """Time-uniform resampling to n points."""
    if len(traj) < 2 or t[-1] - t[0] <= 1e-9:
        return np.repeat(traj[:1], n, axis=0)
    grid = np.linspace(t[0], t[-1], n)
    return np.stack([np.interp(grid, t, traj[:, 0]), np.interp(grid, t, traj[:, 1])], axis=1)


@dataclass(frozen=True)
class MotionPrediction:
    name: str
    confidence: float
    source: str   # "builtin" | "custom"


@dataclass
class MotionResult:
    segment: Segment
    prediction: Optional[MotionPrediction]
    reason: str = ""                       # why nothing fired (for the UI / tuning)
    features: dict = field(default_factory=dict)


# ====================================================================== segmentation
class MotionSegmenter:
    def __init__(self, cfg: MotionSettings) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._prev: Optional[Tuple[float, float, float]] = None  # t, x, y (smoothed)
        self._v = 0.0
        self._recent: Deque[TrackPoint] = deque(maxlen=4)
        self._active: List[TrackPoint] = []
        self._state = "idle"          # idle | moving | cooling
        self._still_since: Optional[float] = None
        self._last_seen: Optional[float] = None

    @property
    def active(self) -> bool:
        return self._state == "moving"

    def trail(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self._active]

    # ------------------------------------------------------------------
    def update(self, p: Optional[TrackPoint], now: float) -> Optional[Segment]:
        cfg = self.cfg
        gap = cfg.gap_ms / 1000.0
        if p is None:
            if self._last_seen is not None:
                lost = now - self._last_seen
                if self._state == "moving" and lost >= gap:
                    return self._finish(self._last_seen)
                if lost > 0.5:                       # hand gone for real: forget smoothing state
                    if self._state == "cooling":
                        self._state = "idle"
                    self._prev, self._v = None, 0.0
                    self._recent.clear()
            return None

        self._last_seen = p.t
        if self._prev is None:
            sx, sy, v_inst = p.x, p.y, 0.0
        else:
            pt, px, py = self._prev
            sx = EMA_ALPHA * p.x + (1 - EMA_ALPHA) * px
            sy = EMA_ALPHA * p.y + (1 - EMA_ALPHA) * py
            dt = p.t - pt
            v_inst = math.hypot(sx - px, sy - py) / dt / p.size if dt > 1e-4 else self._v
        self._prev = (p.t, sx, sy)
        self._v = 0.3 * self._v + 0.7 * v_inst
        sp = TrackPoint(p.t, sx, sy, p.size, p.pose)

        if self._state == "idle":
            self._recent.append(sp)
            if self._v >= cfg.start_speed:
                self._active = list(self._recent)[-3:]
                self._state, self._still_since = "moving", None
        elif self._state == "moving":
            self._active.append(sp)
            if sp.t - self._active[0].t > cfg.max_duration_ms / 1000.0:
                self._active, self._state, self._still_since = [], "cooling", None
            elif self._v < cfg.stop_speed:
                if self._still_since is None:
                    self._still_since = sp.t
                elif sp.t - self._still_since >= gap:
                    return self._finish(self._still_since)
            else:
                self._still_since = None
        else:  # cooling: wait for the over-long movement to stop
            if self._v < cfg.stop_speed:
                if self._still_since is None:
                    self._still_since = sp.t
                elif sp.t - self._still_since >= gap:
                    self._state, self._still_since = "idle", None
                    self._recent.clear()
            else:
                self._still_since = None
        return None

    def _finish(self, end_t: float) -> Optional[Segment]:
        pts = [p for p in self._active if p.t <= end_t]
        self._active, self._state, self._still_since = [], "idle", None
        self._recent.clear()
        if len(pts) < 3:
            return None
        seg = Segment(
            t=np.array([p.t for p in pts]),
            xy=np.array([[p.x, p.y] for p in pts]),
            size=np.array([p.size for p in pts]),
            poses=[p.pose for p in pts],
        )
        if seg.duration * 1000 < self.cfg.min_duration_ms or seg.path_length() < self.cfg.min_path:
            return None
        return seg


# ====================================================================== geometry / built-ins
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def count_reversals(x: np.ndarray, min_swing: float) -> int:
    """Direction changes of x, ignoring swings smaller than min_swing (hysteresis)."""
    if len(x) == 0:
        return 0
    direction, ext, rev = 0, x[0], 0
    for v in x[1:]:
        if direction == 0:
            if abs(v - ext) >= min_swing:
                direction, ext = (1 if v > ext else -1), v
        elif direction > 0:
            if v > ext:
                ext = v
            elif ext - v >= min_swing:
                rev, direction, ext = rev + 1, -1, v
        else:
            if v < ext:
                ext = v
            elif v - ext >= min_swing:
                rev, direction, ext = rev + 1, 1, v
    return rev


def total_turn_deg(traj: np.ndarray, min_step: float = 0.03) -> float:
    d = np.diff(traj, axis=0)
    d = d[np.linalg.norm(d, axis=1) > min_step]
    if len(d) < 3:
        return 0.0
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.degrees(heading[-1] - heading[0]))


def trajectory_features(traj: np.ndarray, duration: float, cfg: MotionSettings) -> dict:
    d = np.diff(traj, axis=0)
    path = float(np.linalg.norm(d, axis=1).sum())
    disp = traj[-1] - traj[0]
    dist = float(np.linalg.norm(disp))
    w, h = (float(np.ptp(traj[:, 0])), float(np.ptp(traj[:, 1])))
    big = max(w, h, 1e-9)
    return {
        "duration": duration, "path": path, "dist": dist, "dx": float(disp[0]), "dy": float(disp[1]),
        "straight": dist / path if path > 1e-9 else 0.0,
        "w": w, "h": h,
        "reversals": count_reversals(traj[:, 0], cfg.wave_min_swing),
        "turn": total_turn_deg(traj),
        "closure": dist / big,
        "aspect": min(w, h) / big,
    }


class BuiltinMotionRecognizer:
    def __init__(self, cfg: MotionSettings) -> None:
        self.cfg = cfg

    def classify_traj(self, traj: np.ndarray, duration: float) -> Tuple[Optional[MotionPrediction], dict]:
        c = self.cfg
        f = trajectory_features(traj, duration, c)
        cands: List[MotionPrediction] = []

        # ---- swipes: long, straight, one dominant axis, quick
        if duration <= c.swipe_max_ms / 1000.0 and f["dist"] >= c.swipe_min_distance and f["straight"] >= c.swipe_straightness:
            ax, ay = abs(f["dx"]), abs(f["dy"])
            name, ratio = None, 0.0
            if ax >= c.swipe_axis_ratio * ay:
                name, ratio = ("swipe_right" if f["dx"] > 0 else "swipe_left"), ax / max(ay, 1e-9)
            elif ay >= c.swipe_axis_ratio * ax:
                name, ratio = ("swipe_down" if f["dy"] > 0 else "swipe_up"), ay / max(ax, 1e-9)
            if name:
                score = min(
                    _clamp01((f["straight"] - c.swipe_straightness) / 0.2),
                    _clamp01((ratio - c.swipe_axis_ratio) / 1.0),
                    _clamp01((f["dist"] - c.swipe_min_distance) / 1.0),
                )
                cands.append(MotionPrediction(name, 0.5 + 0.5 * score, "builtin"))

        # ---- wave: repeated left-right swings, little vertical travel
        if f["reversals"] >= c.wave_min_reversals and f["w"] > 0 and f["h"] <= 0.7 * f["w"]:
            score = min(_clamp01((f["reversals"] - c.wave_min_reversals) / 2.0), _clamp01((0.7 - f["h"] / f["w"]) / 0.3))
            cands.append(MotionPrediction("wave", 0.5 + 0.5 * score, "builtin"))

        # ---- circle: lots of net turning, nearly closed, roughly round
        big = max(f["w"], f["h"])
        if abs(f["turn"]) >= c.circle_min_turn_deg and f["closure"] <= c.circle_closure and f["aspect"] >= 0.5 and big >= 1.0:
            score = min(
                _clamp01((abs(f["turn"]) - c.circle_min_turn_deg) / 90.0),
                _clamp01((c.circle_closure - f["closure"]) / 0.2),
                _clamp01((f["aspect"] - 0.5) / 0.3),
            )
            # heading increases for clockwise motion on screen (image y points down)
            cands.append(MotionPrediction("circle_cw" if f["turn"] > 0 else "circle_ccw", 0.5 + 0.5 * score, "builtin"))

        best = max(cands, key=lambda p: p.confidence, default=None)
        return best, f

    def classify(self, seg: Segment) -> Tuple[Optional[MotionPrediction], dict]:
        return self.classify_traj(seg.trajectory(), seg.duration)


# ====================================================================== DTW
def dtw_batch(templates: np.ndarray, query: np.ndarray, band: int = BAND) -> np.ndarray:
    """DTW distance from `query` (N,2) to every template (T,N,2), vectorised across templates.

    Sakoe-Chiba band; the accumulated cost is divided by N so distances are
    "average hand-lengths of misalignment per step".
    """
    T, N, _ = templates.shape
    cost = np.linalg.norm(templates[:, :, None, :] - query[None, None, :, :], axis=3)  # (T, N, N)
    D = np.full((N + 1, N + 1, T), np.inf)
    D[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(max(1, i - band), min(N, i + band) + 1):
            D[i, j] = cost[:, i - 1, j - 1] + np.minimum(np.minimum(D[i - 1, j], D[i, j - 1]), D[i - 1, j - 1])
    return D[N, N] / N


def augment(sample: np.ndarray, rng: np.random.Generator, n: int) -> List[np.ndarray]:
    """Plausible variants of one recorded motion: scale, slight rotation, speed profile, jitter."""
    N = len(sample)
    u = np.linspace(0.0, 1.0, N)
    out = []
    for _ in range(n):
        s = rng.uniform(0.85, 1.15)
        th = math.radians(rng.uniform(-10, 10))
        g = rng.uniform(0.8, 1.25)
        src = u ** g
        pts = np.stack([np.interp(src, u, sample[:, 0]), np.interp(src, u, sample[:, 1])], axis=1) * s
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        pts = pts @ rot.T + rng.normal(0, 0.02, pts.shape)
        out.append(pts - pts[0])
    return out


# ====================================================================== custom motions
@dataclass
class CustomMotion:
    name: str
    samples: np.ndarray            # (n, N_POINTS, 2)
    pose: str = ""                 # static pose required while moving ('' = any)
    sensitivity: float = 1.0
    created: str = ""

    MIN_SAMPLES = 8
    MAX_SAMPLES = 100

    def validate(self) -> None:
        if not NAME_RE.match(self.name):
            raise ConfigError("Motion name must be 2-32 chars: lowercase letters, digits, underscore; start with a letter")
        if self.name in BUILTIN_MOTIONS or self.name.startswith(("gaze_", "face_")):
            raise ConfigError(f"{self.name!r} is reserved")
        if self.samples.ndim != 3 or self.samples.shape[1:] != (N_POINTS, 2):
            raise ConfigError("Invalid sample shape")
        if not (self.MIN_SAMPLES <= len(self.samples) <= self.MAX_SAMPLES):
            raise ConfigError(f"Need between {self.MIN_SAMPLES} and {self.MAX_SAMPLES} samples (have {len(self.samples)})")
        if not np.isfinite(self.samples).all():
            raise ConfigError("Samples contain invalid numbers")
        if self.pose and not NAME_RE.match(self.pose):
            raise ConfigError("Invalid pose name")
        if not (0.2 <= self.sensitivity <= 5.0):
            raise ConfigError("sensitivity must be between 0.2 and 5.0")


class MotionClassifier:
    """DTW nearest-neighbour over recorded samples + augmented variants.

    State is swapped in one assignment so the GUI thread can reload motions while the
    pipeline thread is classifying.
    """

    FLOOR = 0.30
    CAP = 1.50
    AUG_COPIES = 4

    def __init__(self, cfg: MotionSettings) -> None:
        self.cfg = cfg
        # (motions, templates, base spread)
        self._state: Tuple[Dict[str, CustomMotion], Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float]] = ({}, {}, {})

    def names(self) -> List[str]:
        return list(self._state[0])

    def set_motions(self, motions: Sequence[CustomMotion]) -> None:
        new_m = {m.name: m for m in motions}
        new_t: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        new_b: Dict[str, float] = {}
        for m in motions:
            rng = np.random.default_rng(zlib.crc32(m.name.encode()))
            tpls, owners = [], []
            for i, s in enumerate(m.samples):
                variants = [s] + augment(s, rng, self.AUG_COPIES)
                tpls += variants
                owners += [i] * len(variants)
            tpl, own = np.stack(tpls), np.array(owners)
            new_t[m.name] = (tpl, own)
            nn = [float(dtw_batch(tpl[own != i], s).min()) for i, s in enumerate(m.samples)]  # leave-one-out
            new_b[m.name] = float(np.median(nn))
        self._state = (new_m, new_t, new_b)

    def threshold_for(self, name: str, state=None) -> float:
        motions, _, base = state or self._state
        b = max(self.FLOOR, self.cfg.custom_threshold_scale * base[name])
        return min(self.CAP, b) * motions[name].sensitivity

    def distance(self, name: str, q: np.ndarray, state=None) -> float:
        return float(dtw_batch((state or self._state)[1][name][0], q).min())

    def confidence(self, name: str, q: np.ndarray, state=None) -> float:
        st = state or self._state
        return _clamp01(2.0 * (1.0 - self.distance(name, q, st) / self.threshold_for(name, st)))

    def classify(self, seg: Segment) -> List[MotionPrediction]:
        st = self._state
        motions = st[0]
        if not motions:
            return []
        q = seg.resampled(N_POINTS)
        out = []
        for name, m in motions.items():
            if m.pose and seg.pose_fraction(m.pose) < self.cfg.pose_fraction:
                continue
            c = self.confidence(name, q, st)
            if c > 0:
                out.append(MotionPrediction(name, c, "custom"))
        return out


# ====================================================================== engine
class MotionEngine:
    def __init__(self, cfg: MotionSettings) -> None:
        self.cfg = cfg
        self.segmenter = MotionSegmenter(cfg)
        self.builtin = BuiltinMotionRecognizer(cfg)
        self.custom = MotionClassifier(cfg)
        self._cooldown_until = 0.0
        self.last_result: Optional[MotionResult] = None
        self.last_result_t = 0.0

    def names(self) -> List[str]:
        return list(BUILTIN_MOTIONS) + self.custom.names()

    def reset(self) -> None:
        self.segmenter.reset()

    def update(self, point: Optional[TrackPoint], now: float) -> Tuple[List[GestureEvent], Optional[MotionResult]]:
        seg = self.segmenter.update(point, now)
        if seg is None:
            return [], None
        cfg = self.cfg
        pred: Optional[MotionPrediction] = None
        reason = ""
        features: dict = {}

        customs = [p for p in self.custom.classify(seg) if p.confidence >= cfg.min_confidence]
        if customs:
            pred = max(customs, key=lambda p: p.confidence)
            features = trajectory_features(seg.trajectory(), seg.duration, cfg)
        else:
            bp, features = self.builtin.classify(seg)
            if cfg.require_pose and seg.pose_fraction(cfg.require_pose) < cfg.pose_fraction:
                reason = f"pose is not {cfg.require_pose}"
                if bp:
                    reason += f" (looked like {bp.name})"
            elif bp and bp.confidence >= cfg.min_confidence:
                pred = bp
            else:
                reason = "unrecognized"

        events: List[GestureEvent] = []
        if pred is not None:
            if now < self._cooldown_until:
                reason, pred = f"cooldown (was {pred.name})", None
            else:
                self._cooldown_until = now + cfg.cooldown_ms / 1000.0
                events = [GestureEvent("pressed", pred.name, "motion", now),
                          GestureEvent("released", pred.name, "motion", now)]
        result = MotionResult(seg, pred, reason, features)
        self.last_result, self.last_result_t = result, now
        return events, result
