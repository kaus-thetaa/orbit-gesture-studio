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
    sub.add_parser("hud", help="launch only the small always-on-top orb (Studio opens from its menu)")
    ask = sub.add_parser("ask", help="ask the assistant one question (no GUI)")
    ask.add_argument("question", nargs="+")
    brain = sub.add_parser("brain", help="assistant setup: doctor | set-key")
    brain.add_argument("action", choices=["doctor", "set-key", "bench"])
    brain.add_argument("-n", type=int, default=5, help="bench: requests per model")
    brain.add_argument("--model", help="bench: Gemini model id to test instead of the configured one")
    mem = sub.add_parser("memory", help="shared memory: list | add TEXT | forget TEXT")
    mem.add_argument("action", choices=["list", "add", "forget"])
    mem.add_argument("text", nargs=argparse.REMAINDER)
    mem.add_argument("--private", action="store_true", help="keep this fact local-only (never sent to Gemini)")
    rt = sub.add_parser("routine", help="routines: list | run NAME | add-menu URL")
    rt.add_argument("action", choices=["list", "run", "add-menu"])
    rt.add_argument("arg", nargs="?")
    auto = sub.add_parser("autostart", help="start the orb at login (Linux)")
    auto.add_argument("action", choices=["enable", "disable"])
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
    if cmd == "ask":
        return _cmd_ask(" ".join(args.question))
    if cmd == "brain":
        return _cmd_brain(args.action, args)
    if cmd == "memory":
        return _cmd_memory(args)
    if cmd == "routine":
        return _cmd_routine(args)
    if cmd == "autostart":
        return _cmd_autostart(args.action)
    if cmd == "hud":
        from .gui.app import run_gui
        return run_gui(hud_only=True)
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
        if rep.motion is not None:
            r = rep.motion
            if r.prediction:
                print(f"  motion: {r.prediction.name} ({r.prediction.confidence:.2f}, {r.prediction.source})")
            else:
                f = r.features
                print(f"  motion ignored: {r.reason}  [path {f.get('path', 0):.1f} hl, straight {f.get('straight', 0):.2f}, "
                      f"reversals {f.get('reversals', 0)}, turn {f.get('turn', 0):.0f} deg, {r.segment.duration:.2f} s]")
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


def _make_brain():
    from .brain.core import Brain
    from .settings import Settings
    b = Brain(Settings.load())
    b.net.check()
    return b


def _cmd_ask(question: str) -> int:
    b = _make_brain()
    a = b.ask(question)
    print(a.text)
    print(f"  [{a.source}, {a.latency_ms:.0f} ms{', ' + a.note if a.note else ''}]", file=sys.stderr)
    return 0 if a.source != "error" else 1


def _cmd_brain(action: str, args=None) -> int:
    if action == "set-key":
        import getpass
        from .brain.secrets import secrets_path, set_secret
        key = getpass.getpass("Paste your Gemini API key (input hidden): ").strip()
        if not key:
            print("No key entered.")
            return 1
        print(f"Saved to {set_secret('GEMINI_API_KEY', key)} (permissions 600). It is never written to settings.json or git.")
        return 0
    from .brain.secrets import get_secret
    b = _make_brain()
    if action == "bench":
        return _bench(b, max(1, min(args.n, 20)), args.model)
    c = b.cfg
    print(f"network        : {'online' if b.net.online else 'OFFLINE'}")
    print(f"gemini key     : {'set' if get_secret('GEMINI_API_KEY') else 'NOT SET (python -m orbit brain set-key)'}")
    print(f"gemini model   : {c.gemini_model}")
    if get_secret("GEMINI_API_KEY") and b.net.online:
        try:
            models = b.gemini.list_models()
            print(f"  available    : {', '.join(m for m in models if 'flash' in m)[:300] or '(none with flash)'}")
            if c.gemini_model not in models:
                print("  WARNING      : the configured model is not in this list; ORBIT will auto-pick a flash model, or set brain.gemini_model")
        except Exception as exc:
            print(f"  could not list models: {exc}")
    print(f"ollama         : {c.ollama_host}  model {c.ollama_model}")
    try:
        inst = b.ollama.installed()
        print(f"  installed    : {', '.join(n for n, _ in inst) or '(none: run ollama pull ' + c.ollama_model + ')'}")
        from .brain.providers import pick_ollama_model
        chosen = pick_ollama_model(c.ollama_model, inst)
        if chosen:
            print(f"  will use     : {chosen}" + ("   <-- a code model: poor for general questions; set brain.ollama_model" if "coder" in chosen else ""))
    except Exception as exc:
        print(f"  NOT REACHABLE: {exc}  (install Ollama and run: ollama serve)")
    st = b.status()
    print(f"router         : cloud={st['cloud']} local={st['local']}  gemini requests today {st['gemini_requests_today']}/{st['budget'] or 'unlimited'}")
    print(f"routines       : {st['routines'] or 'none (python -m orbit routine add-menu URL)'}")
    for w in b.warnings:
        print("  PROBLEM:", w)
    return 0


def _bench(b, n: int, gemini_model=None) -> int:
    import statistics

    from .brain.providers import LLMRequest, ProviderError
    req = LLMRequest("Reply with exactly one word.", [("user", "ping")], max_tokens=16)
    if gemini_model:
        b.gemini.model = gemini_model
    for name, prov in (("gemini", b.gemini), ("ollama", b.ollama)):
        if name == "gemini" and not prov.configured():
            print("gemini : no key (python -m orbit brain set-key)")
            continue
        times, err = [], ""
        for _ in range(n):
            try:
                times.append(prov.generate(req).latency_ms)
            except ProviderError as exc:
                err = str(exc)
                break
        if not times:
            print(f"{name:<7}: FAILED - {err}")
            continue
        extra = f", thinking setting #{prov._variant}, {prov.last_attempts} HTTP attempt(s) on the last call" if name == "gemini" else ""
        print(f"{name:<7}: {prov.model}  first {times[0]:.0f} ms | next: " +
              (", ".join(f"{t:.0f}" for t in times[1:]) or "-") + f" ms | median {statistics.median(times):.0f} ms{extra}")
    print("\n'first' includes connecting (TLS); 'next' reuses the connection, which is what the orb does.")
    return 0


def _cmd_memory(args) -> int:
    from .brain.context import ContextStore
    from .errors import ConfigError
    store = ContextStore()
    words = list(args.text)
    private = args.private or "--private" in words
    text = " ".join(w for w in words if w != "--private")
    if args.action == "list":
        for f in store.facts():
            print(f"[{f.scope:<7}] {f.text}")
        return 0
    if args.action == "forget":
        print(f"forgot {store.forget(text)} fact(s)")
        return 0
    try:
        _, created = store.remember(text, "private" if private else "shared")
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("noted" if created else "already known")
    return 0


def _cmd_routine(args) -> int:
    from .brain.routines import parse_routines
    from .settings import Settings
    s = Settings.load()
    if args.action == "add-menu":
        if not args.arg:
            print("usage: python -m orbit routine add-menu URL", file=sys.stderr)
            return 2
        url, kind = args.arg, "menu"
        if "tikmit.com" in url or "coolstuff.work" in url:       # a JS-only viewer; use its public JSON API instead
            url, kind = "https://tikm.coolstuff.work", "menu_api"
            print("Detected the Food Court Menu viewer; using its public API (tikm.coolstuff.work), one small request per refresh.")
        new = {"name": "mess_menu", "kind": kind, "url": url, "refresh_hours": 6}
        _, errors = parse_routines([new])
        if errors:
            print(errors[0], file=sys.stderr)
            return 1
        s.brain.routines = [r for r in s.brain.routines if r.get("name") != "mess_menu"] + [new]
        s.save()
        print("Saved. It will fetch today's menu when you are online (now, and on each startup).")
        return 0
    b = _make_brain()
    if args.action == "list":
        for r in b.scheduler.routines:
            print(f"{r.name}: {r.kind} {r.url} every {r.refresh_hours} h")
        return 0
    res = b.scheduler.run_now(args.arg or "mess_menu")
    print(("OK: " if res.ok else "FAILED: ") + res.message + (f" (parsed by {res.parsed_by})" if res.parsed_by else ""))
    if res.ok:
        import json as _json
        row = b.store.get_doc("mess_menu")
        if row:
            d = _json.loads(row[1])
            for meal, items in (d.get("meals") or {}).items():
                print(f"  {meal:<9} {d.get('times', {}).get(meal, ''):<12} {', '.join(items)}")
    return 0 if res.ok else 1


def _cmd_autostart(action: str) -> int:
    from . import autostart
    try:
        if action == "enable":
            print("Enabled:", autostart.enable())
        else:
            print("Disabled." if autostart.disable() else "Was not enabled.")
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0
