"""Interactive gaze calibration using an OpenCV window (`orbit calibrate-gaze`)."""
from __future__ import annotations

import time

import cv2

from ..settings import Settings
from ..vision.draw import draw_label
from ..vision.face import FaceModule, GazeCalibrator
from .camera import CameraSource


def run(settings: Settings, seconds_per_phase: float = 3.0) -> bool:
    cam = CameraSource(settings.camera)
    face = FaceModule()
    cal = GazeCalibrator()
    cam.start()
    prompts = {"left": "Look at the LEFT edge of your screen", "center": "Look at the CENTRE of your screen",
               "right": "Look at the RIGHT edge of your screen"}
    try:
        t_end = time.monotonic() + 10
        while cam.latest() is None and time.monotonic() < t_end:
            time.sleep(0.05)
        if cam.latest() is None:
            print("Camera did not start.")
            return False
        for phase in GazeCalibrator.PHASES:
            start = time.monotonic()
            while time.monotonic() - start < 2.0 + seconds_per_phase:
                item = cam.latest()
                if item is None:
                    continue
                frame = cv2.flip(item[1], 1)
                warmup = time.monotonic() - start < 2.0
                obs = face.process(frame, int(time.monotonic() * 1000))
                if not warmup:
                    cal.add(phase, obs.gaze_ratio if obs else None)
                view = frame.copy()
                draw_label(view, prompts[phase], (10, 30))
                draw_label(view, "get ready..." if warmup else "hold still - recording", (10, 60),
                           (0, 200, 255) if warmup else (120, 255, 160))
                cv2.imshow("ORBIT gaze calibration (q to cancel)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return False
        left, right = cal.compute()
        settings.gaze.left_threshold, settings.gaze.right_threshold = round(left, 3), round(right, 3)
        settings.save()
        print(f"Saved: gaze.left_threshold={left:.3f}, gaze.right_threshold={right:.3f}")
        print("Set gaze.enabled to true in settings.json and restart ORBIT to use it.")
        return True
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        face.close()
