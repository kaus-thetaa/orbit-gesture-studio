"""Record a custom static gesture: capture samples, review/reject them, live-test, save."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QVBoxLayout,
)

from ..errors import ConfigError
from ..vision.custom_store import GestureRecorder
from ..vision.draw import HAND_CONNECTIONS
from ..vision.gestures import CustomGesture, CustomGestureClassifier

TARGET = 20


def skeleton_icon(sample: np.ndarray, bad: bool = False, size: int = 72) -> QIcon:
    pts = sample.reshape(21, 2)
    pm = QPixmap(size, size)
    pm.fill(QColor(30, 30, 34))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    scale = size / 3.6

    def pos(pt):
        return size / 2 + pt[0] * scale, size * 0.88 - pt[1] * scale

    p.setPen(QPen(QColor(230, 80, 80) if bad else QColor(110, 220, 140), 2))
    for a, b in HAND_CONNECTIONS:
        (x1, y1), (x2, y2) = pos(pts[a]), pos(pts[b])
        p.drawLine(int(x1), int(y1), int(x2), int(y2))
    p.end()
    return QIcon(pm)


class RecorderDialog(QDialog):
    def __init__(self, parent, runtime, bridge) -> None:
        super().__init__(parent)
        self.rt, self.bridge = runtime, bridge
        self.rec = GestureRecorder(runtime.settings.recognition)
        self.clf = CustomGestureClassifier(runtime.settings.recognition)
        self._hand = None
        self.saved_name = None
        self.setWindowTitle("Record custom gesture")
        self.setMinimumSize(780, 560)

        self.name = QLineEdit()
        self.name.setPlaceholderText("gesture name, e.g. rock_sign (lowercase, digits, underscore)")
        self.preview = QLabel("Waiting for camera...")
        self.preview.setMinimumSize(400, 300)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; color:#aaa;")
        self.status = QLabel("")
        self.live = QLabel("Live match: record at least %d samples" % CustomGesture.MIN_SAMPLES)
        self.count = QLabel()
        self.samples = QListWidget()
        self.samples.setViewMode(QListWidget.ViewMode.IconMode)
        self.samples.setIconSize(QSize(72, 72))
        self.samples.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.samples.setMovement(QListWidget.Movement.Static)
        self.samples.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.samples.setMinimumHeight(110)

        self.capture_btn = QPushButton("Capture sample")
        self.auto = QCheckBox(f"Auto-capture (every 0.4 s, up to {TARGET})")
        self.del_btn = QPushButton("Delete selected")
        self.outlier_btn = QPushButton("Flag/remove suspicious")
        self.clear_btn = QPushButton("Clear all")
        self.save_btn = QPushButton("Save gesture")
        self.cancel_btn = QPushButton("Cancel")

        tips = QLabel(
            "Tips: hold the pose and vary it slightly between samples - move closer/farther, tilt a little, "
            "use both hands if you want it to work with either. Samples are stored as landmark numbers only "
            "(no camera images). Hands partly outside the frame or too far away are rejected automatically.")
        tips.setWordWrap(True)
        tips.setStyleSheet("color: gray;")

        top = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self.preview)
        left.addWidget(self.status)
        left.addWidget(self.live)
        top.addLayout(left, 3)
        right = QVBoxLayout()
        right.addWidget(QLabel("Gesture name"))
        right.addWidget(self.name)
        right.addWidget(self.capture_btn)
        right.addWidget(self.auto)
        right.addWidget(tips)
        right.addStretch()
        top.addLayout(right, 2)

        rowbtns = QHBoxLayout()
        for b in (self.del_btn, self.outlier_btn, self.clear_btn):
            rowbtns.addWidget(b)
        rowbtns.addStretch()
        rowbtns.addWidget(self.count)
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(self.cancel_btn)
        bottom.addWidget(self.save_btn)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(QLabel("Samples (select and delete any that look wrong - red = suspicious)"))
        lay.addWidget(self.samples)
        lay.addLayout(rowbtns)
        lay.addLayout(bottom)

        self.capture_btn.clicked.connect(self._capture)
        self.del_btn.clicked.connect(self._delete_selected)
        self.outlier_btn.clicked.connect(self._flag_outliers)
        self.clear_btn.clicked.connect(self._clear)
        self.save_btn.clicked.connect(self._save)
        self.cancel_btn.clicked.connect(self.reject)
        self.bridge.report.connect(self._on_report)
        self.timer = QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self._auto_tick)
        self.timer.start()

        self.rt.mapping_engine.set_suspended(True)   # no actions fire while recording
        self.finished.connect(self._cleanup)
        self._refresh()

    # ------------------------------------------------------------ lifecycle
    def _cleanup(self, _code: int) -> None:
        if getattr(self, "_cleaned", False):
            return
        self._cleaned = True
        self.timer.stop()
        try:
            self.bridge.report.disconnect(self._on_report)
        except (RuntimeError, TypeError):
            pass
        self.rt.mapping_engine.set_suspended(False)

    # ------------------------------------------------------------ live feed
    def _on_report(self, rep) -> None:
        self._hand = rep.hand if rep.reject_reason in (None, "unrecognized") else None
        if rep.frame is not None:
            h, w = rep.frame.shape[:2]
            img = QImage(rep.frame.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.status.setText("Hand OK" if self._hand is not None else f"No usable hand ({rep.reject_reason or 'none'})")
        if self.clf.names() and rep.hand is not None:
            g = self.clf._gestures["preview"]
            c = self.clf.confidence(g, CustomGestureClassifier.features(rep.hand))
            ok = c >= self.rt.settings.recognition.min_gesture_confidence
            self.live.setText(f"Live match: {c:.2f}  {'MATCH' if ok else '-'}")
            self.live.setStyleSheet("color:#3ddc84;font-weight:600;" if ok else "")
        elif self.clf.names():
            self.live.setText("Live match: no hand")

    # ------------------------------------------------------------ samples
    def _capture(self) -> bool:
        reason = self.rec.try_capture(self._hand)
        if reason:
            self.status.setText(f"Rejected: {reason}")
            return False
        self._refresh()
        return True

    def _auto_tick(self) -> None:
        if self.auto.isChecked() and len(self.rec.samples) < TARGET:
            self._capture()

    def _delete_selected(self) -> None:
        rows = sorted({self.samples.row(i) for i in self.samples.selectedItems()}, reverse=True)
        for r in rows:
            self.rec.remove(r)
        self._refresh()

    def _flag_outliers(self) -> None:
        bad = self.rec.outlier_indices()
        if not bad:
            QMessageBox.information(self, "Samples", "No suspicious samples found.")
            return
        if QMessageBox.question(self, "Remove suspicious samples",
                                f"{len(bad)} sample(s) look very different from the rest. Remove them?") == QMessageBox.StandardButton.Yes:
            for i in sorted(bad, reverse=True):
                self.rec.remove(i)
            self._refresh()

    def _clear(self) -> None:
        self.rec.clear()
        self._refresh()

    def _refresh(self) -> None:
        bad = set(self.rec.outlier_indices())
        self.samples.clear()
        for i, s in enumerate(self.rec.samples):
            self.samples.addItem(QListWidgetItem(skeleton_icon(s, i in bad), str(i + 1)))
        n = len(self.rec.samples)
        self.count.setText(f"{n} sample(s) - need {CustomGesture.MIN_SAMPLES}+, ~{TARGET} recommended")
        if n >= CustomGesture.MIN_SAMPLES:
            try:
                g = CustomGesture("preview", self.rec.as_array())
                self.clf.set_gestures([g])
            except Exception:
                self.clf.set_gestures([])
        else:
            self.clf.set_gestures([])
            self.live.setText("Live match: record at least %d samples" % CustomGesture.MIN_SAMPLES)
            self.live.setStyleSheet("")

    # ------------------------------------------------------------ save
    def _save(self) -> None:
        name = self.name.text().strip()
        try:
            gesture = self.rec.build(name)
        except ConfigError as exc:
            QMessageBox.warning(self, "Cannot save gesture", str(exc))
            return
        overlap = self.rec.builtin_overlap()
        if overlap:
            names = ", ".join(f"{k} ({v:.0%})" for k, v in overlap.items())
            if QMessageBox.question(self, "Looks like a built-in gesture",
                                    f"These samples also match the built-in: {names}.\n"
                                    "Custom gestures take priority, so the built-in pose will be hijacked while "
                                    "this custom gesture exists. Save anyway?") != QMessageBox.StandardButton.Yes:
                return
        replace = False
        if self.rt.custom_store.get(gesture.name):
            if QMessageBox.question(self, "Replace?", f"A custom gesture named '{gesture.name}' exists. Replace it?") != QMessageBox.StandardButton.Yes:
                return
            replace = True
        try:
            self.rt.custom_store.add(gesture, replace=replace)
        except ConfigError as exc:
            QMessageBox.warning(self, "Cannot save gesture", str(exc))
            return
        self.rt.reload_custom_gestures()
        self.saved_name = gesture.name
        self.accept()
