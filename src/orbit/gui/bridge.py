"""Qt signals used to hand data from worker threads to the GUI thread safely."""
from PySide6.QtCore import QObject, Signal


class Bridge(QObject):
    report = Signal(object)        # runtime.pipeline.FrameReport
    safety = Signal(bool, str)     # armed, reason
    result = Signal(object)        # dispatcher.ActionResult
    camera = Signal(str)
    focus = Signal(bool, dict)     # active, options
    log_line = Signal(str)

    def attach(self, runtime) -> None:
        runtime.on_report = self.report.emit
        runtime.on_safety = self.safety.emit
        runtime.on_result = self.result.emit
        runtime.on_camera_status = self.camera.emit
        runtime.focus.add_listener(lambda on, opts: self.focus.emit(on, dict(opts)))
