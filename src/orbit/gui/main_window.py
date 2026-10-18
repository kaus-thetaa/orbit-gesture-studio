"""Gesture Studio main window."""
from __future__ import annotations

from dataclasses import fields

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..errors import ConfigError
from ..logging_setup import ring
from .focus_overlay import FocusOverlayManager
from .mapping_dialog import MappingDialog
from .motion_recorder_dialog import MotionRecorderDialog
from .recorder_dialog import RecorderDialog

TUNABLE = [
    ("recognition", "confirm_frames"), ("recognition", "release_frames"), ("recognition", "min_gesture_confidence"),
    ("recognition", "extension_ratio"), ("recognition", "straight_angle_deg"), ("recognition", "thumb_ratio"),
    ("recognition", "min_hand_size"), ("recognition", "custom_distance_scale"),
    ("motion", "start_speed"), ("motion", "stop_speed"), ("motion", "swipe_min_distance"),
    ("motion", "swipe_straightness"), ("motion", "cooldown_ms"), ("motion", "min_confidence"),
    ("motion", "pose_fraction"), ("motion", "custom_threshold_scale"),
]

PILL_OFF = "background:#7a4b00;color:white;padding:6px 12px;border-radius:12px;font-weight:600;"
PILL_ON = "background:#1b7f3b;color:white;padding:6px 12px;border-radius:12px;font-weight:600;"


class MainWindow(QMainWindow):
    def __init__(self, runtime, bridge) -> None:
        super().__init__()
        self.rt, self.bridge = runtime, bridge
        self.overlay = FocusOverlayManager()
        self.setWindowTitle("ORBIT Gesture Studio")
        self.resize(1100, 720)

        # ---- header: safety state is always visible
        self.pill = QLabel()
        self.arm_btn = QPushButton()
        self.estop_btn = QPushButton("EMERGENCY STOP")
        self.estop_btn.setStyleSheet("background:#b00020;color:white;font-weight:700;padding:6px 14px;")
        self.focus_lbl = QLabel("Focus: OFF")
        hk = runtime.settings.safety
        hint = QLabel(f"Global hotkeys: e-stop {hk.emergency_hotkey} | arm/disarm {hk.arm_hotkey} | Esc (in this window) = disarm")
        hint.setStyleSheet("color:gray;")
        header = QHBoxLayout()
        for w in (self.pill, self.arm_btn, self.estop_btn, self.focus_lbl):
            header.addWidget(w)
        header.addStretch()

        tabs = QTabWidget()
        tabs.addTab(self._live_tab(), "Live")
        tabs.addTab(self._studio_tab(), "Gesture Studio")
        tabs.addTab(self._custom_tab(), "Custom gestures")
        tabs.addTab(self._tuning_tab(), "Tuning")
        tabs.addTab(self._log_tab(), "Log")

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addLayout(header)
        lay.addWidget(hint)
        lay.addWidget(tabs)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

        self.arm_btn.clicked.connect(lambda: self.rt.safety.toggle("button"))
        self.estop_btn.clicked.connect(lambda: self.rt.safety.emergency_stop("button"))
        QShortcut(QKeySequence("Esc"), self, activated=lambda: self.rt.safety.disarm("Esc key"))
        bridge.report.connect(self._on_report)
        bridge.safety.connect(self._on_safety)
        bridge.result.connect(self._on_result)
        bridge.camera.connect(lambda s: self.cam_lbl.setText(s))
        bridge.focus.connect(self._on_focus)
        bridge.log_line.connect(self.log.appendPlainText)
        ring.listeners.append(bridge.log_line.emit)
        self._on_safety(self.rt.safety.armed, "startup")
        self._reload_table()
        self._reload_custom()
        for w in self.rt.warnings:
            self.log.appendPlainText("WARNING: " + w)

    # ================================================================ tabs
    def _live_tab(self) -> QWidget:
        w = QWidget()
        self.video = QLabel("Starting camera...")
        self.video.setMinimumSize(640, 480)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background:#111;color:#aaa;")
        self.cam_lbl, self.fps_lbl, self.hand_lbl = QLabel("-"), QLabel("-"), QLabel("-")
        self.pred_lbl, self.stable_lbl, self.active_lbl = QLabel("-"), QLabel("-"), QLabel("-")
        self.motion_state_lbl, self.motion_lbl = QLabel("-"), QLabel("-")
        self.motion_lbl.setWordWrap(True)
        form = QFormLayout()
        for name, lbl in (("Camera", self.cam_lbl), ("FPS / latency", self.fps_lbl), ("Hand", self.hand_lbl),
                          ("Raw gesture", self.pred_lbl), ("Confirmed gesture", self.stable_lbl),
                          ("Motion", self.motion_state_lbl), ("Last motion", self.motion_lbl),
                          ("Active mappings", self.active_lbl)):
            form.addRow(name, lbl)
        self.events = QListWidget()
        side = QVBoxLayout()
        side.addLayout(form)
        side.addWidget(QLabel("Recent events  (while DISARMED these show what *would* happen)"))
        side.addWidget(self.events)
        lay = QHBoxLayout(w)
        lay.addWidget(self.video, 3)
        lay.addLayout(side, 2)
        return w

    def _studio_tab(self) -> QWidget:
        w = QWidget()
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["On", "Gesture", "Action", "Mode", "Parameters", "Cooldown"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._toggle_enabled)
        self.table.itemDoubleClicked.connect(lambda _i: self._edit())
        add, edit, delete = QPushButton("Add mapping"), QPushButton("Edit"), QPushButton("Delete")
        test, rec = QPushButton("Test action now (3 s)"), QPushButton("Record static gesture...")
        rec_motion = QPushButton("Record motion gesture...")
        rec_motion.clicked.connect(self._record_motion)
        add.clicked.connect(self._add)
        edit.clicked.connect(self._edit)
        delete.clicked.connect(self._delete)
        test.clicked.connect(self._test)
        rec.clicked.connect(self._record)
        row = QHBoxLayout()
        for b in (add, edit, delete, test):
            row.addWidget(b)
        row.addStretch()
        row.addWidget(rec)
        row.addWidget(rec_motion)
        info = QLabel("Mappings are saved automatically. Tick/untick 'On' to enable/disable. The Live tab "
                      "(while disarmed) lets you test gestures without running any action.")
        info.setWordWrap(True)
        info.setStyleSheet("color:gray;")
        lay = QVBoxLayout(w)
        lay.addWidget(self.table)
        lay.addLayout(row)
        lay.addWidget(info)
        return w

    def _custom_tab(self) -> QWidget:
        w = QWidget()
        self.custom_list, self.motion_list = QListWidget(), QListWidget()
        rec, delete = QPushButton("Record static gesture..."), QPushButton("Delete selected")
        rec_m, delete_m = QPushButton("Record motion gesture..."), QPushButton("Delete selected")
        rec.clicked.connect(self._record)
        delete.clicked.connect(self._delete_custom)
        rec_m.clicked.connect(self._record_motion)
        delete_m.clicked.connect(self._delete_motion)
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Static poses (stored as landmark numbers only, no images)"))
        lay.addWidget(self.custom_list)
        row = QHBoxLayout()
        row.addWidget(rec); row.addWidget(delete); row.addStretch()
        lay.addLayout(row)
        lay.addWidget(QLabel("Motion gestures (stored as trajectory shapes only). Built-ins: swipe_left/right/up/down, wave, circle_cw/ccw"))
        lay.addWidget(self.motion_list)
        row2 = QHBoxLayout()
        row2.addWidget(rec_m); row2.addWidget(delete_m); row2.addStretch()
        lay.addLayout(row2)
        return w

    def _tuning_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout()
        for section, name in TUNABLE:
            obj = getattr(self.rt.settings, section)
            f = next(f for f in fields(obj) if f.name == name)
            val = getattr(obj, name)
            lo, hi = f.metadata.get("min", 0), f.metadata.get("max", 100)
            if isinstance(val, int):
                spin = QSpinBox()
                spin.setRange(int(lo), int(hi))
            else:
                spin = QDoubleSpinBox()
                spin.setRange(float(lo), float(hi))
                spin.setDecimals(2)
                spin.setSingleStep(0.05)
            spin.setValue(val)
            spin.setToolTip(f.metadata.get("doc", ""))
            spin.valueChanged.connect(lambda v, o=obj, n=name: setattr(o, n, v))
            form.addRow(name if section == "recognition" else f"{section}.{name}", spin)
        cam = QSpinBox()
        cam.setRange(0, 10)
        cam.setValue(self.rt.settings.camera.index)
        cam.valueChanged.connect(lambda v: setattr(self.rt.settings.camera, "index", v))
        restart = QPushButton("Restart camera")
        restart.clicked.connect(lambda: self.rt.camera.restart())
        save = QPushButton("Save settings to disk")
        save.clicked.connect(self._save_settings)
        self.landmarks = QCheckBox("Draw hand landmarks")
        self.landmarks.setChecked(self.rt.settings.ui.show_landmarks)
        self.landmarks.toggled.connect(lambda v: setattr(self.rt.settings.ui, "show_landmarks", v))
        form.addRow("camera index", cam)
        form.addRow("", restart)
        form.addRow("", self.landmarks)
        form.addRow("", save)
        note = QLabel("Changes apply immediately but are only kept after 'Save'. See docs/CALIBRATION.md.\n"
                      "Gaze / face signals are experimental and configured in settings.json (docs/GAZE_AND_FACE.md).")
        note.setStyleSheet("color:gray;")
        lay = QVBoxLayout(w)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addStretch()
        return w

    def _log_tab(self) -> QWidget:
        w = QWidget()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setPlainText("\n".join(ring.lines[-200:]))
        lay = QVBoxLayout(w)
        lay.addWidget(self.log)
        return w

    # ================================================================ slots
    def _on_report(self, rep) -> None:
        if rep.frame is not None:
            h, w = rep.frame.shape[:2]
            img = QImage(rep.frame.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
            self.video.setPixmap(QPixmap.fromImage(img).scaled(
                self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.video.setText(f"No camera frames ({rep.camera_status}). Trying to reconnect...")
        self.cam_lbl.setText(rep.camera_status)
        self.fps_lbl.setText(f"{rep.fps:.1f} fps / {rep.inference_ms:.0f} ms")
        self.hand_lbl.setText("tracking" if rep.hand is not None and not rep.reject_reason else (rep.reject_reason or "-"))
        self.pred_lbl.setText(f"{rep.prediction.name} ({rep.prediction.confidence:.2f})" if rep.prediction else "-")
        self.stable_lbl.setText(rep.stable_gesture or "-")
        self.active_lbl.setText(", ".join(self.rt.mapping_engine.active_ids()) or "-")
        self.motion_state_lbl.setText("tracking a movement..." if rep.motion_active else "idle")
        if rep.motion is not None:
            r = rep.motion
            if r.prediction:
                self.motion_lbl.setText(f"{r.prediction.name} ({r.prediction.confidence:.2f})")
            else:
                f = r.features
                detail = (f"path {f.get('path', 0):.1f} hl, straight {f.get('straight', 0):.2f}, "
                          f"reversals {f.get('reversals', 0)}, turn {f.get('turn', 0):.0f} deg") if f else ""
                self.motion_lbl.setText(f"none: {r.reason}  [{detail}]")
                self._event(f"  motion ignored: {r.reason}")
        for ev in rep.events:
            if ev.source == "motion" and ev.kind == "released":
                continue
            self._event(f"{ev.kind.upper()} {ev.name}")
        for d in rep.decisions:
            note = "" if d.executed else f"  [not executed: {d.reason}]"
            self._event(f"  -> {d.phase} {d.action}{note}")

    def _event(self, text: str) -> None:
        self.events.insertItem(0, text)
        while self.events.count() > 60:
            self.events.takeItem(self.events.count() - 1)

    def _on_safety(self, armed: bool, reason: str) -> None:
        self.pill.setText("● ARMED - actions are LIVE" if armed else "● DISARMED - dry run, no actions")
        self.pill.setStyleSheet(PILL_ON if armed else PILL_OFF)
        self.arm_btn.setText("Disarm" if armed else "Arm actions")
        self.statusBar().showMessage(f"{'Armed' if armed else 'Disarmed'} ({reason})")

    def _on_result(self, res) -> None:
        if not res.ok:
            self.statusBar().showMessage(f"Action failed: {res.error}", 8000)
            self.log.appendPlainText(f"ACTION FAILED {res.request.action}: {res.error}")

    def _on_focus(self, active: bool, options: dict) -> None:
        self.focus_lbl.setText("Focus: ON" if active else "Focus: OFF")
        self.focus_lbl.setStyleSheet("color:#1b7f3b;font-weight:700;" if active else "")
        self.overlay.set_active(active, options)

    # ================================================================ mappings
    def _reload_table(self) -> None:
        self.table.blockSignals(True)
        mappings = self.rt.store.all()
        self.table.setRowCount(len(mappings))
        for r, m in enumerate(mappings):
            on = QTableWidgetItem()
            on.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            on.setCheckState(Qt.CheckState.Checked if m.enabled else Qt.CheckState.Unchecked)
            on.setData(Qt.ItemDataRole.UserRole, m.id)
            self.table.setItem(r, 0, on)
            for c, text in enumerate([m.gesture, m.action, m.mode, ", ".join(f"{k}={v}" for k, v in m.params.items()),
                                      f"{m.cooldown_ms} ms"], start=1):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.blockSignals(False)

    def _selected_id(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Mappings", "Select a mapping first.")
            return None
        return self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)

    def _toggle_enabled(self, item) -> None:
        if item.column() != 0:
            return
        try:
            self.rt.store.set_enabled(item.data(Qt.ItemDataRole.UserRole), item.checkState() == Qt.CheckState.Checked)
        except ConfigError as exc:
            QMessageBox.warning(self, "Cannot enable", str(exc))
        self._reload_table()

    def _add(self, gesture=None) -> None:
        if MappingDialog(self, self.rt, gesture=gesture if isinstance(gesture, str) else None).exec():
            self._reload_table()

    def _edit(self) -> None:
        mid = self._selected_id()
        if mid and MappingDialog(self, self.rt, self.rt.store.get(mid)).exec():
            self._reload_table()

    def _delete(self) -> None:
        mid = self._selected_id()
        if mid and QMessageBox.question(self, "Delete", f"Delete mapping '{mid}'?") == QMessageBox.StandardButton.Yes:
            self.rt.store.remove(mid)
            self._reload_table()

    def _test(self) -> None:
        mid = self._selected_id()
        if not mid:
            return
        m = self.rt.store.get(mid)
        if QMessageBox.question(self, "Run action now?",
                                f"This will run '{m.action}' immediately (manual test), on whatever window is in front.\nContinue?"
                                ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.statusBar().showMessage(self.rt.test_mapping(mid), 6000)
        except ConfigError as exc:
            QMessageBox.warning(self, "Test failed", str(exc))

    # ================================================================ custom gestures
    def _reload_custom(self) -> None:
        self.custom_list.clear()
        for g in self.rt.custom_store.gestures:
            self.custom_list.addItem(f"{g.name}  ({len(g.samples)} samples)")
        self.motion_list.clear()
        for m in self.rt.motion_store.motions:
            self.motion_list.addItem(f"{m.name}  ({len(m.samples)} examples, pose: {m.pose or 'any'})")

    def _record(self) -> None:
        dlg = RecorderDialog(self, self.rt, self.bridge)
        if dlg.exec() and dlg.saved_name:
            self._reload_custom()
            if QMessageBox.question(self, "Assign an action?",
                                    f"Gesture '{dlg.saved_name}' saved. Assign an action to it now?") == QMessageBox.StandardButton.Yes:
                self._add(dlg.saved_name)

    def _record_motion(self) -> None:
        dlg = MotionRecorderDialog(self, self.rt, self.bridge)
        if dlg.exec() and dlg.saved_name:
            self._reload_custom()
            if QMessageBox.question(self, "Assign an action?",
                                    f"Motion '{dlg.saved_name}' saved. Assign an action to it now?") == QMessageBox.StandardButton.Yes:
                self._add(dlg.saved_name)

    def _delete_motion(self) -> None:
        row = self.motion_list.currentRow()
        if row < 0:
            return
        name = self.rt.motion_store.motions[row].name
        used = [m.id for m in self.rt.store.all() if m.gesture == name]
        msg = f"Delete custom motion '{name}'?" + (f"\nMappings using it ({', '.join(used)}) will stop firing." if used else "")
        if QMessageBox.question(self, "Delete", msg) == QMessageBox.StandardButton.Yes:
            self.rt.motion_store.remove(name)
            self.rt.reload_custom_motions()
            self._reload_custom()

    def _delete_custom(self) -> None:
        row = self.custom_list.currentRow()
        if row < 0:
            return
        name = self.rt.custom_store.gestures[row].name
        used = [m.id for m in self.rt.store.all() if m.gesture == name]
        msg = f"Delete custom gesture '{name}'?" + (f"\nMappings using it ({', '.join(used)}) will stop firing." if used else "")
        if QMessageBox.question(self, "Delete", msg) == QMessageBox.StandardButton.Yes:
            self.rt.custom_store.remove(name)
            self.rt.reload_custom_gestures()
            self._reload_custom()

    def _save_settings(self) -> None:
        self.rt.settings.save()
        self.statusBar().showMessage("Settings saved", 4000)

    def closeEvent(self, event) -> None:
        self.overlay.set_active(False, {})
        try:
            ring.listeners.remove(self.bridge.log_line.emit)
        except ValueError:
            pass
        super().closeEvent(event)
