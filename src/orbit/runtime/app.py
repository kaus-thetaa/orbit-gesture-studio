"""OrbitRuntime: constructs and wires every component. Used by both the GUI and the CLI."""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Callable, List, Optional

from ..actions.base import ActionContext
from ..actions.builtin import build_default_registry
from ..actions.focus import FocusController
from ..actions.audio import default_audio_backend
from ..actions.keyboard import PynputKeyboard
from ..actions.platform import detect_platform
from ..dispatcher import ActionDispatcher, ActionRequest, ActionResult
from ..errors import ConfigError, ModelError
from ..mapping import MappingStore
from ..safety import HotkeyManager, SafetyController
from ..settings import Settings
from ..vision.custom_store import CustomGestureStore
from ..vision.motion import BUILTIN_MOTIONS, MotionEngine
from ..vision.motion_store import MotionStore
from ..vision.face import FaceSignalHub
from ..vision.lips import MouthTracker
from ..vision.gestures import CustomGestureClassifier, FingerPatternRecognizer, GestureRegistry
from ..vision.stabilizer import GestureStabilizer
from .camera import CameraSource
from .mapping_engine import MappingEngine
from .pipeline import FrameReport, Pipeline

log = logging.getLogger(__name__)


class OrbitRuntime:
    def __init__(self, settings: Optional[Settings] = None, keyboard: Any = None, audio: Any = None) -> None:
        self.settings = settings or Settings.load()
        s = self.settings
        self.warnings: List[str] = []

        self.focus = FocusController()
        self.actions = build_default_registry()
        self.platform = detect_platform(s.safety.platform_shortcuts)
        self.ctx = ActionContext(keyboard or PynputKeyboard(), self.focus, self.platform,
                                 tuple(s.safety.shortcut_denylist), audio or default_audio_backend())
        self.safety = SafetyController(armed=s.safety.start_armed)
        if s.safety.start_armed:
            log.warning("start_armed is true: actions are live from startup")
        self.dispatcher = ActionDispatcher(self.actions, self.ctx, self.safety, on_result=self._on_result)
        self.motion = MotionEngine(s.motion)
        self.motion_store = MotionStore()
        self.store = MappingStore(self.actions, denylist=lambda: self.settings.safety.shortcut_denylist,
                                  motion_names=self.motion_names)
        self.store.load()
        self.warnings += self.store.load_errors
        self.mapping_engine = MappingEngine(self.store, self.dispatcher, self.safety)

        self.custom_store = CustomGestureStore()
        self.custom_clf = CustomGestureClassifier(s.recognition)
        self.builtin_rec = FingerPatternRecognizer(s.recognition)
        self.gestures = GestureRegistry(s.recognition)
        self.gestures.register(self.builtin_rec)
        self.gestures.register(self.custom_clf)
        self.reload_custom_gestures()
        self.reload_custom_motions()

        self.camera = CameraSource(s.camera, on_status=self._on_camera_status)
        self.stabilizer = GestureStabilizer(s.recognition)
        self.face_hub = FaceSignalHub(s.gaze, s.face)
        self.tracker: Any = None
        self.face_module: Any = None
        self.pipeline: Optional[Pipeline] = None
        self.lips: Optional[MouthTracker] = None
        self._brain: Any = None
        self.hotkeys: Optional[HotkeyManager] = None
        self._timers: List[threading.Timer] = []

        # callbacks for the UI layer (invoked from worker threads!)
        self.on_report: Optional[Callable[[FrameReport], None]] = None
        self.on_result: Optional[Callable[[ActionResult], None]] = None
        self.on_safety: Optional[Callable[[bool, str], None]] = None
        self.on_camera_status: Optional[Callable[[str], None]] = None
        self.on_activity: Optional[Callable[[list, list], None]] = None
        self.on_ask_hotkey: Optional[Callable[[], None]] = None

        self.safety.add_listener(self._on_safety_change)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start camera + vision. Raises ModelError if the hand model cannot be obtained."""
        from ..vision.hand_tracker import HandTracker

        self.tracker = HandTracker(self.settings.recognition)
        if self.face_hub.wants_face or self.settings.lips.enabled:
            try:
                from ..vision.face import FaceModule
                self.face_module = FaceModule()
                if self.settings.lips.enabled:
                    self.lips = MouthTracker(self.settings.lips, self.face_module.landmarks_px)
            except Exception as exc:  # optional features must never block the core app
                msg = f"Gaze/face/lips features disabled: {exc}"
                log.error(msg)
                self.warnings.append(msg)
                self.settings.gaze.enabled = False
                self.settings.face.enabled = False
                self.settings.lips.enabled = False
                self.face_hub.rebuild()
        if sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE") == "wayland":
            msg = ("Wayland session detected: synthetic key events and global hotkeys are blocked, so keyboard/snap/"
                   "desktop actions and emergency hotkeys will not work. Use an X11 session (see docs/TROUBLESHOOTING.md).")
            log.warning(msg)
            self.warnings.append(msg)
        self.pipeline = Pipeline(
            self.settings, self.camera, self.tracker, self.gestures, self.stabilizer, self.mapping_engine,
            face_module=self.face_module, face_hub=self.face_hub, motion=self.motion, lips=self.lips,
            on_report=lambda r: self.on_report and self.on_report(r),
            on_activity=lambda e, d: self.on_activity and self.on_activity(e, d),
        )
        self.dispatcher.start()
        if self.settings.brain.enabled:
            self.brain.start()
        self.camera.start()
        self.pipeline.start()
        if self.settings.safety.enable_hotkeys:
            bindings = {
                self.settings.safety.emergency_hotkey: lambda: self.safety.emergency_stop("emergency hotkey"),
                self.settings.safety.arm_hotkey: lambda: self.safety.toggle("arm hotkey"),
            }
            if self.on_ask_hotkey and self.settings.hud.ask_hotkey:
                bindings[self.settings.hud.ask_hotkey] = lambda: self.on_ask_hotkey and self.on_ask_hotkey()
            self.hotkeys = HotkeyManager(bindings)
            if not self.hotkeys.start():
                self.warnings.append(f"Global hotkeys unavailable ({self.hotkeys.error}); use the on-screen buttons.")

    def stop(self) -> None:
        self.safety.disarm("shutdown")
        for t in self._timers:
            t.cancel()
        if self.pipeline:
            self.pipeline.stop()
        self.camera.stop()
        self.dispatcher.join_idle(1.0)
        self.dispatcher.stop()
        if self.hotkeys:
            self.hotkeys.stop()
        if self._brain is not None:
            self._brain.stop()
        for closable in (self.tracker, self.face_module):
            if closable is not None:
                closable.close()

    # ------------------------------------------------------------------ helpers
    @property
    def brain(self):
        """The assistant, created on first use (keeps tests and headless gesture-only runs light)."""
        if self._brain is None:
            from ..brain.core import Brain
            self._brain = Brain(self.settings)
        return self._brain

    def reload_custom_gestures(self) -> None:
        self.custom_clf.set_gestures(self.custom_store.load())

    def reload_custom_motions(self) -> None:
        self.motion.custom.set_motions(self.motion_store.load())

    def motion_names(self) -> List[str]:
        return list(BUILTIN_MOTIONS) + self.motion.custom.names()

    def known_gestures(self) -> List[str]:
        names = list(self.builtin_rec.names()) + self.custom_clf.names() + self.motion_names()
        names += ["finger_on_lips", "gaze_left", "gaze_right"]
        names += [f"face_{sig.get('name')}" for sig in self.settings.face.signals if isinstance(sig, dict) and sig.get("name")]
        return names

    def test_mapping(self, mapping_id: str, hold_seconds: float = 3.0) -> str:
        """Run a mapping's action once, on explicit user request (works while disarmed)."""
        m = self.store.get(mapping_id)
        if m is None:
            raise ConfigError(f"No mapping {mapping_id!r}")
        action = self.actions.get(m.action)
        req = ActionRequest(m.id, m.action, "activate", dict(m.params), m.mode)
        if not self.dispatcher.submit(req, force=True):
            return "Could not queue the action (queue full)."
        if action.stateful:
            t = threading.Timer(hold_seconds, lambda: self.dispatcher.submit(
                ActionRequest(m.id, m.action, "deactivate", dict(m.params), m.mode), force=True))
            t.daemon = True
            t.start()
            self._timers.append(t)
            return f"Ran '{m.action}' for {hold_seconds:.0f}s as a manual test."
        return f"Ran '{m.action}' once as a manual test."

    # ------------------------------------------------------------------ callbacks
    def _on_result(self, result: ActionResult) -> None:
        if self.on_result:
            self.on_result(result)

    def _on_camera_status(self, status: str) -> None:
        if self.on_camera_status:
            self.on_camera_status(status)

    def _on_safety_change(self, armed: bool, reason: str) -> None:
        if not armed:
            self.dispatcher.purge()                 # drop queued, not-yet-started activations
            self.mapping_engine.release_all(reason)  # undo whatever is currently on
            self.focus.deactivate()
        if self.on_safety:
            self.on_safety(armed, reason)
