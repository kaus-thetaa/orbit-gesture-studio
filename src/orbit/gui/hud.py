"""The always-on-top orb: tiny, translucent, out of the way. Click to ask; answers appear as one line.

Colours: green = cloud model ready, amber = local model only, grey = no model available.
Right-click: open Gesture Studio / quit. The ask box can also be opened with a global hotkey (X11).
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QRadialGradient
from PySide6.QtWidgets import QLineEdit, QMenu, QWidget

GREEN, AMBER, GREY, RED = QColor(80, 220, 140), QColor(245, 180, 60), QColor(140, 140, 150), QColor(235, 90, 90)


class InputPopup(QLineEdit):
    submitted = Signal(str)

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setPlaceholderText("ask ORBIT...")
        self.setFixedWidth(340)
        self.setStyleSheet("QLineEdit{background:rgba(24,24,28,235);color:#eee;border:1px solid #444;border-radius:12px;padding:8px 12px;font-size:14px;}")
        self.returnPressed.connect(self._go)

    def _go(self) -> None:
        text = self.text().strip()
        self.clear()
        self.hide()
        if text:
            self.submitted.emit(text)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.clear()
            self.hide()
        else:
            super().keyPressEvent(e)

    def focusOutEvent(self, e) -> None:
        super().focusOutEvent(e)
        QTimer.singleShot(0, self.hide)


class Hud(QWidget):
    answered = Signal(str, str)        # text, source (from worker threads)
    notice = Signal(str)               # routine events, e.g. "Menu updated"
    ask_requested = Signal()           # global hotkey

    def __init__(self, brain, cfg, open_studio=None, quit_cb=None, disarm_cb=None) -> None:
        super().__init__(None)
        self.brain, self.cfg = brain, cfg
        self.open_studio, self.quit_cb, self.disarm_cb = open_studio, quit_cb, disarm_cb
        self.text, self.color = "", GREY
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setWindowOpacity(cfg.idle_opacity)
        self.input = InputPopup()
        self.input.submitted.connect(self.ask)
        self.answered.connect(self._on_answer)
        self.notice.connect(lambda m: self.show_text(m, "routine", 3.0))
        self.ask_requested.connect(self.open_input)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._clear_text)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self.refresh_status)
        self._status_timer.start(3000)
        self._relayout()
        self.refresh_status()

    # ---------------------------------------------------------------- layout
    def _font(self) -> QFont:
        return QFont("Sans Serif", 10)

    def _relayout(self) -> None:
        d = self.cfg.orb_size
        pad = 12
        if self.text:
            fm = QFontMetrics(self._font())
            screen = QGuiApplication.primaryScreen().availableGeometry()
            shown = fm.elidedText(self.text, Qt.TextElideMode.ElideRight, int(screen.width() * 0.6))
            self._shown = shown
            bubble_w = fm.horizontalAdvance(shown) + 2 * pad
        else:
            self._shown, bubble_w = "", 0
        w = d + (bubble_w + 8 if bubble_w else 0) + 4
        h = max(d, 30) + 4
        self.resize(w, h)
        self._place()

    def _place(self) -> None:
        g = QGuiApplication.primaryScreen().availableGeometry()
        m = self.cfg.margin
        right = "right" in self.cfg.corner
        bottom = "bottom" in self.cfg.corner
        x = g.right() - self.width() - m + 1 if right else g.left() + m
        y = g.bottom() - self.height() - m + 1 if bottom else g.top() + m
        self.move(x, y)

    def orb_center(self) -> QPoint:
        d = self.cfg.orb_size
        return QPoint(self.width() - d // 2 - 2, self.height() // 2)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        d = self.cfg.orb_size
        c = self.orb_center()
        if self._shown:
            fm = QFontMetrics(self._font())
            bw = fm.horizontalAdvance(self._shown) + 24
            rect = QRectF(0, 2, bw, self.height() - 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(22, 22, 26, 225))
            p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
            p.setPen(QColor(238, 238, 242))
            p.setFont(self._font())
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._shown)
        g = QRadialGradient(c.x() - d * 0.15, c.y() - d * 0.2, d * 0.7)
        g.setColorAt(0.0, self.color.lighter(150))
        g.setColorAt(1.0, self.color.darker(130))
        p.setPen(QColor(255, 255, 255, 70))
        p.setBrush(g)
        p.drawEllipse(c, d // 2 - 1, d // 2 - 1)

    # ---------------------------------------------------------------- behaviour
    def enterEvent(self, _e) -> None:
        self.setWindowOpacity(1.0)

    def leaveEvent(self, _e) -> None:
        self.setWindowOpacity(self.cfg.idle_opacity if not self.text else 0.95)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.open_input()
        elif e.button() == Qt.MouseButton.RightButton:
            self._menu(e.globalPosition().toPoint())

    def _menu(self, pos: QPoint) -> None:
        m = QMenu(self)
        for label, cb in (("Open Gesture Studio", self.open_studio), ("Disarm actions", self.disarm_cb), ("Quit ORBIT", self.quit_cb)):
            if cb:
                a = QAction(label, m)
                a.triggered.connect(lambda _=False, f=cb: f())
                m.addAction(a)
        m.exec(pos)

    def open_input(self) -> None:
        g = self.geometry()
        self.input.move(max(0, g.right() - self.input.width() + 1), max(0, g.top() - 52))
        self.input.show()
        self.input.raise_()
        self.input.activateWindow()
        self.input.setFocus()

    def ask(self, text: str) -> None:
        self.show_text("...", "wait", 30.0)
        self.brain.ask_async(text, lambda a: self.answered.emit(a.text, a.source))

    def _on_answer(self, text: str, source: str) -> None:
        self.color = {"gemini": GREEN, "ollama": AMBER, "error": RED}.get(source, self.color)
        self.show_text(text or "...", source, self.cfg.answer_seconds)

    def show_text(self, text: str, source: str, seconds: float) -> None:
        self.text = text
        self._relayout()
        self.setWindowOpacity(0.95)
        self.update()
        self._hide_timer.start(int(seconds * 1000))

    def _clear_text(self) -> None:
        self.text = ""
        self._relayout()
        self.setWindowOpacity(self.cfg.idle_opacity)
        self.update()

    def refresh_status(self) -> None:
        try:
            st = self.brain.status()
        except Exception:
            return
        self.color = GREEN if st["cloud"] == "ok" else AMBER if st["local"] == "ok" else GREY
        self.update()
