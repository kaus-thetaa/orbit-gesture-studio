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
    sub.add_parser("download-models", help="download the MediaPipe hand model")
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
        from .models import ensure_model

        try:
            print(f"hand_landmarker: {ensure_model('hand_landmarker')}")
        except ModelError as exc:
            print(exc, file=sys.stderr)
            return 2
        return 0
    if cmd == "check-config":
        return _check_config()
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
    print("settings.json: OK")
    print(f"mappings.json: {len(mappings)} valid mapping(s)")
    for mapping in mappings:
        print(
            f"  [{'on ' if mapping.enabled else 'off'}] "
            f"{mapping.gesture:<14} -> {mapping.action:<18} ({mapping.mode})"
        )
    for error in store.load_errors:
        print("  PROBLEM:", error)
    return 1 if store.load_errors else 0


def _run_headless(args) -> int:
    import cv2

    from .errors import ModelError
    from .runtime.app import OrbitRuntime

    runtime = OrbitRuntime()
    runtime.focus.add_listener(lambda on, opts: print(f"*** FOCUS MODE {'ON' if on else 'OFF'} ***"))
    last = {"line": "", "frame": None}

    def on_report(report):
        line = (
            f"{report.camera_status:<12} gesture={report.prediction.name if report.prediction else '-':<14} "
            f"stable={report.stable_gesture or '-':<14} fps={report.fps:4.1f} {report.reject_reason or ''}"
        )
        if report.frame is not None:
            last["frame"] = report.frame  # OpenCV windows are not thread-safe.
        if line != last["line"] and not args.preview:
            last["line"] = line
            print(line)
        if report.motion is not None:
            result = report.motion
            if result.prediction:
                print(
                    f"  motion: {result.prediction.name} "
                    f"({result.prediction.confidence:.2f}, {result.prediction.source})"
                )
            else:
                features = result.features
                print(
                    f"  motion ignored: {result.reason}  "
                    f"[path {features.get('path', 0):.1f} hl, "
                    f"straight {features.get('straight', 0):.2f}, "
                    f"reversals {features.get('reversals', 0)}, "
                    f"turn {features.get('turn', 0):.0f} deg, {result.segment.duration:.2f} s]"
                )
        for decision in report.decisions:
            print(
                f"  -> {decision.phase} {decision.action} "
                f"[{decision.mapping_id}] executed={decision.executed} {decision.reason}"
            )

    runtime.on_report = on_report
    try:
        runtime.start()
    except ModelError as exc:
        print(exc, file=sys.stderr)
        return 2
    for warning in runtime.warnings:
        print("warning:", warning)
    if args.armed:
        runtime.safety.arm("cli --armed")
    state = "ARMED" if runtime.safety.armed else "in DRY-RUN mode (use --armed to execute actions)"
    print("ORBIT running", state, "- Ctrl+C to quit")
    print(f"Emergency stop hotkey: {runtime.settings.safety.emergency_hotkey}")
    end_time = time.monotonic() + args.seconds if args.seconds else None
    try:
        while end_time is None or time.monotonic() < end_time:
            if args.preview:
                if last["frame"] is not None:
                    cv2.imshow("ORBIT (q quit, e emergency stop)", last["frame"])
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("e"):
                    runtime.safety.emergency_stop("preview key")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        cv2.destroyAllWindows()
    return 0
