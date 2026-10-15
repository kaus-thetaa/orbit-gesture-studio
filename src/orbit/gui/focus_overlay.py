"""Focus mode visuals: a click-through vignette over every screen + a small pill label.

It never intercepts mouse/keyboard input and never touches other windows.
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget


class _Overlay(QWidget):
    def __init__(self, screen, dim: int) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.dim = dim
        self.setGeometry(screen.geometry())

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        radius = (r.width() ** 2 + r.height() ** 2) ** 0.5 / 2
        g = QRadialGradient(r.center(), radius)
        g.setColorAt(0.0, QColor(0, 0, 0, 0))
        g.setColorAt(0.55, QColor(0, 0, 0, 0))
        g.setColorAt(1.0, QColor(0, 0, 0, int(255 * self.dim / 100)))
        p.fillRect(r, g)
        text = "FOCUS"
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        w, h = 84, 26
        pill = QRectF((r.width() - w) / 2, 10, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 20, 24, 200))
        p.drawRoundedRect(pill, 13, 13)
        p.setPen(QColor(150, 255, 190))
        p.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)


class FocusOverlayManager:
    def __init__(self) -> None:
        self._overlays: List[_Overlay] = []

    def set_active(self, active: bool, options: dict) -> None:
        self._clear()
        if not active:
            return
        dim = int(options.get("dim", 35))
        for screen in QGuiApplication.screens():
            ov = _Overlay(screen, dim)
            ov.show()
            self._overlays.append(ov)

    def _clear(self) -> None:
        for ov in self._overlays:
            ov.close()
            ov.deleteLater()
        self._overlays = []
