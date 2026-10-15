"""Vision pipeline: camera frame -> hand -> gesture -> stabiliser -> mappings.

Runs on its own thread. Action execution happens on the dispatcher's thread, so a slow
or failing action can never stall frame processing.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import cv2
import numpy as np

from ..settings import Settings
from ..vision.draw import draw_hand, draw_label
from ..vision.face import FaceSignalHub
from ..vision.gestures import GestureRegistry
from ..vision.quality import check_hand_quality
from ..vision.stabilizer import GestureStabilizer
from ..vision.types import GestureEvent, GesturePrediction, HandObservation
from .mapping_engine import Decision, MappingEngine

log = logging.getLogger(__name__)


@dataclass
class FrameReport:
    frame: Optional[np.ndarray] = None            # annotated BGR (mirrored); None if no camera
    hand: Optional[HandObservation] = None
    reject_reason: Optional[str] = None           # why no usable hand: no_hand / partial_hand / hand_too_small / unrecognized
    prediction: Optional[GesturePrediction] = None
    stable_gesture: Optional[str] = None
    events: List[GestureEvent] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    fps: float = 0.0
    inference_ms: float = 0.0
    camera_status: str = "stopped"
    gaze_ratio: Optional[float] = None


class Pipeline:
    def __init__(self, settings: Settings, camera: Any, tracker: Any, gestures: GestureRegistry,
                 stabilizer: GestureStabilizer, mapping_engine: MappingEngine,
                 face_module: Any = None, face_hub: Optional[FaceSignalHub] = None,
                 on_report: Optional[Callable[[FrameReport], None]] = None,
                 on_activity: Optional[Callable[[List[GestureEvent], List[Decision]], None]] = None) -> None:
        self.settings, self.camera, self.tracker = settings, camera, tracker
        self.gestures, self.stabilizer, self.engine = gestures, stabilizer, mapping_engine
        self.face_module, self.face_hub = face_module, face_hub
        self.on_report, self.on_activity = on_report, on_activity
        self.annotate = True
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fps = 0.0
        self._last_frame_t = 0.0
        self._lost = False

    # ------------------------------------------------------------ one frame (also used by tests)
    def process_frame(self, frame: np.ndarray, ts_ms: int) -> FrameReport:
        """`frame` must already be mirrored (selfie view)."""
        t0 = time.perf_counter()
        hand = self.tracker.process(frame, ts_ms)
        reject: Optional[str] = None
        pred: Optional[GesturePrediction] = None
        if hand is None:
            reject = "no_hand"
        else:
            reject = check_hand_quality(hand, self.settings.recognition)
            if reject is None:
                pred = self.gestures.classify(hand)
                if pred is None:
                    reject = "unrecognized"

        events = self.stabilizer.update(pred.name if pred else None)
        gaze_ratio = None
        if self.face_hub is not None and self.face_module is not None and self.face_hub.wants_face:
            try:
                obs = self.face_module.process(frame, ts_ms)
            except Exception:
                log.exception("face processing failed")
                obs = None
            more, dbg = self.face_hub.update(obs)
            events += more
            gaze_ratio = dbg.get("gaze_ratio")

        decisions: List[Decision] = []
        for ev in events:
            decisions += self.engine.handle(ev)
            log.info("%s %s (%s)", ev.kind, ev.name, ev.source)
        if (events or decisions) and self.on_activity:
            self.on_activity(events, decisions)

        now = time.perf_counter()
        dt = now - self._last_frame_t if self._last_frame_t else 0.0
        self._last_frame_t = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

        view = None
        if self.annotate:
            view = frame.copy()
            if hand is not None and self.settings.ui.show_landmarks:
                draw_hand(view, hand, ok=reject in (None, "unrecognized"))
            label = pred.name if pred else (reject or "")
            if pred:
                label += f" {pred.confidence:.2f}"
            draw_label(view, label)
            if self.stabilizer.active:
                draw_label(view, f"ACTIVE: {self.stabilizer.active}", org=(10, 58), color=(120, 255, 160))

        return FrameReport(
            frame=view, hand=hand, reject_reason=reject, prediction=pred,
            stable_gesture=self.stabilizer.active, events=events, decisions=decisions,
            fps=self._fps, inference_ms=(time.perf_counter() - t0) * 1000.0,
            camera_status=getattr(self.camera, "status", "streaming"), gaze_ratio=gaze_ratio,
        )

    def handle_camera_lost(self) -> FrameReport:
        """No usable frames: release everything held, once."""
        if not self._lost:
            self._lost = True
            log.warning("camera feed lost - releasing held gestures")
            events = self.stabilizer.reset()
            if self.face_hub:
                events += self.face_hub.reset()
            decisions: List[Decision] = []
            for ev in events:
                decisions += self.engine.handle(ev)
            decisions += self.engine.release_all("camera lost")
            if self.on_activity and (events or decisions):
                self.on_activity(events, decisions)
        return FrameReport(camera_status=getattr(self.camera, "status", "unknown"), reject_reason="camera_lost")

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="orbit-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _emit(self, report: FrameReport) -> None:
        if self.on_report:
            try:
                self.on_report(report)
            except Exception:
                log.exception("report callback failed")

    def _run(self) -> None:
        last_seq = -1
        last_emit = 0.0
        errors = 0
        while not self._stop.is_set():
            item = self.camera.latest()
            if item is None:
                rep = self.handle_camera_lost()
                self._emit(rep)
                self._stop.wait(0.2)
                continue
            seq, raw = item
            if seq == last_seq:
                self._stop.wait(0.003)
                continue
            last_seq = seq
            self._lost = False
            try:
                report = self.process_frame(cv2.flip(raw, 1), int(time.monotonic() * 1000))
                errors = 0
            except Exception:
                errors += 1
                if errors in (1, 10, 100):
                    log.exception("frame processing failed (%d in a row)", errors)
                self._stop.wait(0.02)
                continue
            interval = 1.0 / max(1, self.settings.ui.preview_fps)
            if time.monotonic() - last_emit >= interval:
                last_emit = time.monotonic()
                self._emit(report)
            elif report.events or report.decisions:
                self._emit(report)  # never drop reports that carry events
