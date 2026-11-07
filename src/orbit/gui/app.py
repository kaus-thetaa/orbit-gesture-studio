"""GUI entry point."""
from __future__ import annotations

import logging
import sys
import threading

log = logging.getLogger(__name__)


def _ensure_models(parent=None) -> bool:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

    from ..models import ensure_model, model_path

    if model_path("hand_landmarker").exists():
        return True
    dlg = QProgressDialog("Downloading the MediaPipe hand-tracking model (first run only)...", None, 0, 0, parent)
    dlg.setWindowTitle("ORBIT setup")
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.show()
    result: dict = {}

    def work() -> None:
        try:
            ensure_model("hand_landmarker")
        except Exception as exc:
            result["err"] = exc

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        QApplication.processEvents()
        t.join(0.05)
    dlg.close()
    if "err" in result:
        QMessageBox.critical(None, "Model download failed", str(result["err"]))
        return False
    return True


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from ..runtime.app import OrbitRuntime
    from .bridge import Bridge
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("ORBIT Gesture Studio")
    if not _ensure_models():
        return 2
    runtime = OrbitRuntime()
    bridge = Bridge()
    bridge.attach(runtime)
    win = MainWindow(runtime, bridge)
    try:
        runtime.start()
    except Exception as exc:
        log.exception("startup failed")
        QMessageBox.critical(None, "ORBIT could not start", f"{type(exc).__name__}: {exc}\n\nRun `python -m orbit doctor` for diagnostics.")
        return 2
    for w in runtime.warnings:
        win.log.appendPlainText("WARNING: " + w)
    win.show()
    rc = app.exec()
    runtime.stop()
    return rc
