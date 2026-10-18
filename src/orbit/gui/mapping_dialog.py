"""Add / edit a gesture -> action mapping."""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QMessageBox, QSpinBox, QVBoxLayout,
)

from ..errors import ConfigError
from ..mapping import Mapping, make_id


class MappingDialog(QDialog):
    def __init__(self, parent, runtime, mapping: Optional[Mapping] = None, gesture: Optional[str] = None) -> None:
        super().__init__(parent)
        self.rt, self.original = runtime, mapping
        self.setWindowTitle("Edit mapping" if mapping else "New mapping")
        self.setMinimumWidth(420)

        self.gesture = QComboBox()
        self.gesture.setEditable(True)
        self.gesture.addItems(runtime.known_gestures())
        self.action = QComboBox()
        for name in runtime.actions.names():
            self.action.addItem(name)
        self.mode = QComboBox()
        self.enabled = QCheckBox("Enabled")
        self.enabled.setChecked(True)
        self.cooldown = QSpinBox()
        self.cooldown.setRange(0, 10_000)
        self.cooldown.setSuffix(" ms")
        self.cooldown.setValue(500)
        self.help = QLabel()
        self.help.setWordWrap(True)
        self.help.setStyleSheet("color: gray;")
        self.param_form = QFormLayout()
        self.param_widgets: Dict[str, object] = {}

        form = QFormLayout()
        form.addRow("Gesture", self.gesture)
        form.addRow("Action", self.action)
        form.addRow("Activation", self.mode)
        form.addRow("", self.help)
        form.addRow("Cooldown", self.cooldown)
        form.addRow("", self.enabled)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("Action parameters"))
        lay.addLayout(self.param_form)
        lay.addWidget(buttons)

        self.action.currentTextChanged.connect(self._action_changed)
        self.gesture.currentTextChanged.connect(lambda _t: self._refresh_modes())
        if mapping:
            self.gesture.setCurrentText(mapping.gesture)
            self.action.setCurrentText(mapping.action)
            self.enabled.setChecked(mapping.enabled)
            self.cooldown.setValue(mapping.cooldown_ms)
        elif gesture:
            self.gesture.setCurrentText(gesture)
        self._action_changed(self.action.currentText())
        if mapping:
            self.mode.setCurrentText(mapping.mode)
            self._load_params(mapping.params)

    def _refresh_modes(self) -> None:
        action = self.rt.actions.get(self.action.currentText())
        modes = list(action.modes)
        if self.gesture.currentText().strip() in set(self.rt.motion_names()):
            modes = [m for m in modes if m != "hold"]
        keep = self.mode.currentText()
        self.mode.clear()
        self.mode.addItems(modes)
        if keep in modes:
            self.mode.setCurrentText(keep)

    # ---------------------------------------------------------------- dynamic form
    def _action_changed(self, name: str) -> None:
        action = self.rt.actions.get(name)
        self._refresh_modes()
        self.help.setText(action.description + (
            "  Modes: hold = active while the gesture is held; toggle = each confirmed gesture flips on/off; "
            "trigger = fire once. Motion gestures (swipes, wave, circles, recorded motions) are momentary: trigger or toggle only."))
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self.param_widgets.clear()
        for key, spec in action.params.items():
            if spec.type == "int":
                w = QSpinBox()
                w.setRange(int(spec.min if spec.min is not None else -10**6), int(spec.max if spec.max is not None else 10**6))
                w.setValue(int(spec.default if spec.default is not None else 0))
            else:
                w = QLineEdit()
                if spec.default:
                    w.setText(str(spec.default))
                w.setPlaceholderText(spec.doc)
            w.setToolTip(spec.doc)
            self.param_widgets[key] = w
            self.param_form.addRow(key, w)
        if not action.params:
            self.param_form.addRow(QLabel("(none)"))

    def _load_params(self, params: dict) -> None:
        for key, value in params.items():
            w = self.param_widgets.get(key)
            if isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QLineEdit):
                w.setText(str(value))

    def _collect(self) -> dict:
        out = {}
        for key, w in self.param_widgets.items():
            out[key] = w.value() if isinstance(w, QSpinBox) else w.text()
        return out

    # ---------------------------------------------------------------- save
    def _save(self) -> None:
        gesture = self.gesture.currentText().strip()
        action = self.action.currentText()
        try:
            if self.original:
                m = Mapping(self.original.id, gesture, action, self.mode.currentText(), self._collect(),
                            self.enabled.isChecked(), self.cooldown.value())
                self.rt.store.update(self.original.id, m)
            else:
                mid = make_id(gesture, action, [x.id for x in self.rt.store.all()])
                m = Mapping(mid, gesture, action, self.mode.currentText(), self._collect(),
                            self.enabled.isChecked(), self.cooldown.value())
                self.rt.store.add(m)
        except ConfigError as exc:
            QMessageBox.warning(self, "Cannot save mapping", str(exc))
            return
        self.accept()
