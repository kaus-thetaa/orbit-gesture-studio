"""Record a custom MOTION gesture: perform it repeatedly, review/reject examples, live-test, save."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QRadioButton, QVBoxLayout,
)

from ..errors import ConfigError
from ..vision.motion import CustomMotion, MotionClassifier
from ..vision.motion_store import MotionRecorder

TARGET = 25


def trajectory_icon(traj: np.ndarray, bad: bool = False, size: int = 72) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(30, 30, 34))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    lo, hi = traj.min(axis=0), traj.max(axis=0)
    span = max(float((hi - lo).max()), 1e-6)
    pts = (traj - (lo + hi) / 2) / span * (size * 0.72) + size / 2
    p.setPen(QPen(QColor(230, 80, 80) if bad else QColor(110, 220, 140), 2))
    for a, b in zip(pts[:-1], pts[1:]):
        p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
    p.setBrush(QColor(255, 255, 255))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(int(pts[0][0]) - 3, int(pts[0][1]) - 3, 6, 6)          # start dot
    p.setBrush(QColor(255, 200, 0))
    p.drawEllipse(int(pts[-1][0]) - 4, int(pts[-1][1]) - 4, 8, 8)        # end dot
    p.end()
    return QIcon(pm)


class MotionRecorderDialog(QDialog):
    def __init__(self, parent, runtime, bridge) -> None:
        super().__init__(parent)
        self.rt, self.bridge = runtime, bridge
        self.rec = MotionRecorder(runtime.settings.motion)
        self.clf = MotionClassifier(runtime.settings.motion)
        self.saved_name = None
        self._cleaned = False
        self.setWindowTitle("Record motion gesture")
        self.setMinimumSize(800, 600)

        self.name = QLineEdit()
        self.name.setPlaceholderText("motion name, e.g. zigzag (lowercase, digits, underscore)")
        self.preview = QLabel("Waiting for camera...")
        self.preview.setMinimumSize(400, 300)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; color:#aaa;")
        self.status = QLabel("Ready: perform the motion, then pause for a moment.")
        self.last = QLabel("")
        self.last.setWordWrap(True)
        self.pose_lbl = QLabel("")
        self.count = QLabel()
        self.record_rb = QRadioButton("Record examples")
        self.test_rb = QRadioButton("Test (nothing is recorded)")
        self.record_rb.setChecked(True)
        self.pose_chk = QCheckBox("Require the same hand pose while moving (recommended: fewer false triggers)")
        self.pose_chk.setChecked(True)
        self.samples = QListWidget()
        self.samples.setViewMode(QListWidget.ViewMode.IconMode)
        self.samples.setIconSize(QSize(72, 72))
        self.samples.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.samples.setMovement(QListWidget.Movement.Static)
        self.samples.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.samples.setMinimumHeight(110)
        self.del_btn, self.flag_btn = QPushButton("Delete selected"), QPushButton("Flag/remove suspicious")
        self.clear_btn, self.save_btn, self.cancel_btn = QPushButton("Clear all"), QPushButton("Save motion"), QPushButton("Cancel")

        tips = QLabel(
            "How: keep your hand in view, make the motion, then pause ~0.3 s. Each completed movement becomes one "
            f"example. Record 20-30 examples, varying speed, size and position a little (min {CustomMotion.MIN_SAMPLES}). "
            "Delete any that look wrong (white dot = start, yellow = end). Then switch to Test and check it matches "
            "while other movements don't. Examples are saved as shapes only (no camera images).")
        tips.setWordWrap(True)
        tips.setStyleSheet("color: gray;")

        left = QVBoxLayout()
        for w in (self.preview, self.status, self.last):
            left.addWidget(w)
        right = QVBoxLayout()
        for w in (QLabel("Motion name"), self.name, self.record_rb, self.test_rb, self.pose_lbl, self.pose_chk, tips):
            right.addWidget(w)
        right.addStretch()
        top = QHBoxLayout()
        top.addLayout(left, 3)
        top.addLayout(right, 2)
        rowbtns = QHBoxLayout()
        for b in (self.del_btn, self.flag_btn, self.clear_btn):
            rowbtns.addWidget(b)
        rowbtns.addStretch()
        rowbtns.addWidget(self.count)
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(self.cancel_btn)
        bottom.addWidget(self.save_btn)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(QLabel("Examples (red = looks different from the rest)"))
        lay.addWidget(self.samples)
        lay.addLayout(rowbtns)
        lay.addLayout(bottom)

        self.del_btn.clicked.connect(self._delete_selected)
        self.flag_btn.clicked.connect(self._flag_outliers)
        self.clear_btn.clicked.connect(self._clear)
        self.save_btn.clicked.connect(self._save)
        self.cancel_btn.clicked.connect(self.reject)
        self.bridge.report.connect(self._on_report)
        self.rt.mapping_engine.set_suspended(True)      # no actions fire while recording
        self.finished.connect(self._cleanup)
        self._refresh()

    def _cleanup(self, _code: int) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        try:
            self.bridge.report.disconnect(self._on_report)
        except (RuntimeError, TypeError):
            pass
        self.rt.mapping_engine.set_suspended(False)

    # ------------------------------------------------------------ live feed
    def _on_report(self, rep) -> None:
        if rep.frame is not None:
            h, w = rep.frame.shape[:2]
            img = QImage(rep.frame.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        if rep.motion_active:
            self.status.setText("Tracking movement... (finish and pause)")
        elif rep.motion is None:
            self.status.setText("Ready: perform the motion, then pause for a moment.")
        if rep.motion is None:
            return
        seg = rep.motion.segment
        if self.record_rb.isChecked():
            reason = self.rec.add_segment(seg)
            if reason:
                self.last.setText(f"Rejected: {reason}")
            else:
                self.last.setText(f"Captured example {len(self.rec.samples)}  ({seg.duration:.2f} s, pose: {seg.dominant_pose() or 'mixed'})")
                self._refresh()
        else:
            self._test(seg)

    def _test(self, seg) -> None:
        if not self.clf.names():
            self.last.setText(f"Record at least {CustomMotion.MIN_SAMPLES} examples first.")
            return
        q = seg.resampled()
        d = self.clf.distance("preview", q)
        thr = self.clf.threshold_for("preview")
        c = self.clf.confidence("preview", q)
        ok = c >= self.rt.settings.motion.min_confidence
        self.last.setText(f"{'MATCH' if ok else 'no match'}  confidence {c:.2f}  (distance {d:.2f} / threshold {thr:.2f})")
        self.last.setStyleSheet("color:#3ddc84;font-weight:600;" if ok else "")

    # ------------------------------------------------------------ samples
    def _delete_selected(self) -> None:
        for r in sorted({self.samples.row(i) for i in self.samples.selectedItems()}, reverse=True):
            self.rec.remove(r)
        self._refresh()

    def _flag_outliers(self) -> None:
        bad = self.rec.outlier_indices()
        if not bad:
            QMessageBox.information(self, "Examples", "No suspicious examples found.")
            return
        if QMessageBox.question(self, "Remove suspicious examples",
                                f"{len(bad)} example(s) look very different from the rest. Remove them?") == QMessageBox.StandardButton.Yes:
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
            self.samples.addItem(QListWidgetItem(trajectory_icon(s, i in bad), str(i + 1)))
        n = len(self.rec.samples)
        self.count.setText(f"{n} example(s) - need {CustomMotion.MIN_SAMPLES}+, {TARGET}-30 recommended")
        pose = self.rec.majority_pose()
        self.pose_lbl.setText(f"Detected pose while moving: {pose or 'none / mixed'}")
        if n >= CustomMotion.MIN_SAMPLES:
            try:
                self.clf.set_motions([CustomMotion("preview", self.rec.as_array(), pose=pose if self.pose_chk.isChecked() else "")])
            except Exception:
                self.clf.set_motions([])
        else:
            self.clf.set_motions([])

    # ------------------------------------------------------------ save
    def _save(self) -> None:
        name = self.name.text().strip()
        try:
            motion = self.rec.build(name, require_pose=self.pose_chk.isChecked())
        except ConfigError as exc:
            QMessageBox.warning(self, "Cannot save motion", str(exc))
            return
        if self.rt.custom_store.get(motion.name):
            QMessageBox.warning(self, "Name in use", f"'{motion.name}' is already a custom static gesture. Pick another name.")
            return
        overlap = self.rec.builtin_overlap()
        if overlap:
            names = ", ".join(f"{k} ({v:.0%})" for k, v in overlap.items())
            if QMessageBox.question(self, "Looks like a built-in motion",
                                    f"These examples are also recognised as the built-in: {names}.\nCustom motions take priority, "
                                    "so this would replace that built-in while it exists. Save anyway?") != QMessageBox.StandardButton.Yes:
                return
        replace = False
        if self.rt.motion_store.get(motion.name):
            if QMessageBox.question(self, "Replace?", f"A custom motion named '{motion.name}' exists. Replace it?") != QMessageBox.StandardButton.Yes:
                return
            replace = True
        try:
            self.rt.motion_store.add(motion, replace=replace)
        except ConfigError as exc:
            QMessageBox.warning(self, "Cannot save motion", str(exc))
            return
        self.rt.reload_custom_motions()
        self.saved_name = motion.name
        self.accept()
