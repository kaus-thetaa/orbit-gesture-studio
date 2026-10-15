"""Command-line entry point: `python -m orbit [command]`."""
from __future__ import annotations

import argparse
import sys
import time

from . import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="orbit", description="ORBIT Gesture Studio")
    parser.add_argument("--version", action="version", version=f"orbit {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="launch Gesture Studio (default)")
    run = sub.add_parser("run", help="headless mode: print recognised gestures; optional preview window")
    run.add_argument("--armed", action="store_true", help="actually execute actions (default: dry run)")
    run.add_argument("--preview", action="store_true", help="show an OpenCV preview window (q to quit, e = emergency stop)")
    run.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    sub.add_parser("doctor", help="print an environment report")
    cams = sub.add_parser("cameras", help="probe camera indices")
    cams.add_argument("--max-index", type=int, default=4)
    dl = sub.add_parser("download-models", help="download the MediaPipe model files")
    dl.add_argument("--face", action="store_true", help="also download the face landmarker (gaze/face features)")
    sub.add_parser("calibrate-gaze", help="EXPERIMENTAL: calibrate gaze thresholds")
    sub.add_parser("check-config", help="validate settings.json and mappings.json")
    args = parser.parse_args(argv)

    from .logging_setup import setup_logging

    setup_logging(args.log_level)
    cmd = args.cmd or "gui"

    if cmd == "doctor":
        from . import diagnostics
        print("\n".join(diagnostics.collect()))
        return 0
    if cmd == "cameras":
        from .runtime.camera import list_cameras
        from .settings import Settings
        for row in list_cameras(args.max_index, Settings.load().camera):
            print(row)
        print("\nSet camera.index in settings.json to the index marked ok=True.")
        return 0
    if cmd == "download-models":
        from .errors import ModelError
        from .models import MODELS, ensure_model
        names = ["hand_landmarker"] + (["face_landmarker"] if args.face else [])
        try:
            for n in names:
                print(f"{n}: {ensure_model(n)}")
        except ModelError as exc:
            print(exc, file=sys.stderr)
            return 2
        return 0
    if cmd == "check-config":
        return _check_config()
    if cmd == "calibrate-gaze":
        from .runtime import calibrate
        from .settings import Settings
        return 0 if calibrate.run(Settings.load()) else 1
    if cmd == "run":
        return _run_headless(args)
    from .gui.app import run_gui
    return run_gui()


def _check_config() -> int:
    from .actions.builtin import build_default_registry
    from .mapping import MappingStore
    from .settings import Settings

    settings = Settings.load()
    store = MappingStore(build_default_registry(), denylist=lambda: settings.safety.shortcut_denylist)
    mappings = store.load()
    print(f"settings.json: OK")
    print(f"mappings.json: {len(mappings)} valid mapping(s)")
    for m in mappings:
        print(f"  [{'on ' if m.enabled else 'off'}] {m.gesture:<14} -> {m.action:<18} ({m.mode})")
    for err in store.load_errors:
        print("  PROBLEM:", err)
    return 1 if store.load_errors else 0


def _run_headless(args) -> int:
    import cv2

    from .errors import ModelError
    from .runtime.app import OrbitRuntime

    rt = OrbitRuntime()
    rt.focus.add_listener(lambda on, opts: print(f"*** FOCUS MODE {'ON' if on else 'OFF'} ***"))
    last = {"line": "", "frame": None}

    def on_report(rep):
        line = f"{rep.camera_status:<12} gesture={rep.prediction.name if rep.prediction else '-':<14} stable={rep.stable_gesture or '-':<14} fps={rep.fps:4.1f} {rep.reject_reason or ''}"
        if rep.frame is not None:
            last["frame"] = rep.frame  # drawn on the main thread (OpenCV windows are not thread-safe)
        if line != last["line"] and not args.preview:
            last["line"] = line
            print(line)
        for d in rep.decisions:
            print(f"  -> {d.phase} {d.action} [{d.mapping_id}] executed={d.executed} {d.reason}")

    rt.on_report = on_report
    try:
        rt.start()
    except ModelError as exc:
        print(exc, file=sys.stderr)
        return 2
    for w in rt.warnings:
        print("warning:", w)
    if args.armed:
        rt.safety.arm("cli --armed")
    print("ORBIT running", "ARMED" if rt.safety.armed else "in DRY-RUN mode (use --armed to execute actions)", "- Ctrl+C to quit")
    print(f"Emergency stop hotkey: {rt.settings.safety.emergency_hotkey}")
    t_end = time.monotonic() + args.seconds if args.seconds else None
    try:
        while t_end is None or time.monotonic() < t_end:
            if args.preview:
                if last["frame"] is not None:
                    cv2.imshow("ORBIT (q quit, e emergency stop)", last["frame"])
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("e"):
                    rt.safety.emergency_stop("preview key")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rt.stop()
        cv2.destroyAllWindows()
    return 0
