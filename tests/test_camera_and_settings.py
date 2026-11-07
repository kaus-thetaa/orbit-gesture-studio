import time

import numpy as np

from orbit.runtime.camera import CameraSource
from orbit.settings import CameraSettings, Settings


class FakeCap:
    def __init__(self, frames_before_fail=None):
        self.n = 0
        self.limit = frames_before_fail
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.002)
        self.n += 1
        if self.limit is not None and self.n > self.limit:
            return False, None
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        self.released = True


def wait_for(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_camera_streams_and_stops():
    cap = FakeCap()
    cam = CameraSource(CameraSettings(), opener=lambda cfg: cap)
    cam.start()
    assert wait_for(lambda: cam.latest() is not None)
    assert cam.status == "streaming"
    cam.stop()
    assert cap.released and cam.status == "stopped"


def test_camera_disconnect_and_reconnect():
    caps = []

    def opener(cfg):
        cap = FakeCap(frames_before_fail=5 if not caps else None)
        caps.append(cap)
        return cap

    cfg = CameraSettings(reconnect_interval_s=0.5, read_failure_limit=3)
    statuses = []
    cam = CameraSource(cfg, opener=opener, on_status=statuses.append)
    cam.start()
    assert wait_for(lambda: "reconnecting" in statuses)
    assert wait_for(lambda: len(caps) >= 2 and cam.status == "streaming")
    assert caps[0].released
    cam.stop()


def test_camera_open_failure_retries():
    attempts = []

    def opener(cfg):
        attempts.append(1)
        return None if len(attempts) < 3 else FakeCap()

    cam = CameraSource(CameraSettings(reconnect_interval_s=0.5), opener=opener)
    cam.start()
    assert wait_for(lambda: cam.latest() is not None, timeout=5)
    assert len(attempts) >= 3
    cam.stop()


def test_stale_frames_count_as_lost():
    cap = FakeCap()
    cam = CameraSource(CameraSettings(stale_after_s=0.2), opener=lambda c: cap)
    cam.start()
    assert wait_for(lambda: cam.latest() is not None)
    cam._stop.set()
    cam._thread.join()
    time.sleep(0.35)
    assert cam.latest() is None


def test_settings_defaults_are_sane():
    s = Settings()
    assert s.safety.start_armed is False
    assert s.motion.enabled is True


def test_settings_load_save_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    s = Settings.load(p)
    s.recognition.confirm_frames = 9
    s.save(p)
    assert Settings.load(p).recognition.confirm_frames == 9


def test_settings_tolerates_garbage(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"recognition": {"confirm_frames": "lots", "release_frames": -5}, "camera": 7, "bogus": 1}')
    s = Settings.load(p)
    assert s.recognition.confirm_frames == 6 and s.recognition.release_frames == 1
    p.write_text("not json")
    assert Settings.load(p).recognition.confirm_frames == 6
    assert (tmp_path / "s.json.corrupt").exists()
