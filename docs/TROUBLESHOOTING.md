# Troubleshooting

First run `python -m orbit doctor` and keep the output handy.

## Camera

| Symptom | Fix |
|---|---|
| Black preview / "reconnecting" forever | `python -m orbit cameras` and set `camera.index` in `settings.json` (or Tuning tab) to an index with `ok=True`. |
| Works in other apps but not ORBIT | Only one app can own the webcam. Close Teams/Zoom/browser tabs/Camera app. |
| Windows blocks camera | Settings → Privacy & security → Camera → allow **desktop apps**. Check the laptop's physical privacy shutter / `Fn` camera key. |
| Very slow start (10+ s) | Set `camera.backend` to `"dshow"` (fast) or `"msmf"` in `settings.json`. |
| Low FPS | Try `camera.fourcc: "MJPG"` (default) or `""`; lower `camera.width/height`; close other camera apps. |

## Install

| Error | Fix |
|---|---|
| `No matching distribution found for mediapipe` | Your Python is too new/old for the wheel. Use Python 3.11 or 3.12: `py -3.11 -m venv .venv`. |
| `ImportError: DLL load failed` (cv2/PySide6) | Install the *Microsoft Visual C++ Redistributable (x64)*. |
| `cv2` import weirdness, two OpenCV packages | `pip uninstall -y opencv-python opencv-python-headless` then `pip install --force-reinstall opencv-contrib-python`. |
| `numpy` version conflicts | `pip install --upgrade mediapipe numpy` in a fresh venv. |
| PowerShell: "running scripts is disabled" | `Set-ExecutionPolicy -Scope Process Bypass` then activate the venv again. |
| Model download fails (proxy/firewall) | Download the URL printed in the error in a browser and save it to the path printed. |

## Actions

| Symptom | Fix |
|---|---|
| Gesture detected but nothing happens | Are you **armed**? (header pill). Check the Live event list for `[not executed: ...]` and the Log tab. |
| Shortcuts ignored in some apps | Windows blocks injection into elevated (Admin) windows unless ORBIT runs elevated too. |
| `snap`/`show desktop` does nothing on Linux | Needs X11 (not Wayland) and a desktop that uses those shortcuts; override in `safety.platform_shortcuts`. |
| macOS: nothing happens | System Settings → Privacy & Security → Accessibility → allow your terminal/Python. |
| Global hotkeys unavailable | Use the on-screen buttons; on macOS grant Input Monitoring. |
| Focus overlay not visible | Focus mode is a vignette at the screen edges (centre stays clear). Increase `dim`. |

## Linux / Debian

| Symptom | Fix |
|---|---|
| Actions, snapping, hotkeys do nothing; `doctor` says `wayland` | Wayland blocks synthetic input. Log out, click the gear on the login screen, pick **GNOME on Xorg** (or your desktop's X11 session). Check with `echo $XDG_SESSION_TYPE` → `x11`. |
| `externally-managed-environment` from pip | Don't use system pip; use the venv (`python3 -m venv .venv`, needs `sudo apt install python3-venv`). |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1` |
| Camera index found but `Permission denied` | `sudo usermod -aG video $USER`, then log out and back in. |
| No `/dev/video*` | Webcam disabled in BIOS/Fn key, or driver missing. Try `v4l2-ctl --list-devices` (package `v4l-utils`) and `lsusb`. |
| Two `/dev/videoN` per camera | Normal: the first is usually the capture node, the second metadata. Try `camera.index` 0, then 2. |
| `pip` can't find a mediapipe wheel | Check `python3 --version`; use 3.11 or 3.12. On Debian 13 (Python 3.13) install a 3.12 via `pyenv` or `uv python install 3.12`. |
| Show desktop / overview / snap use the wrong shortcut | Defaults assume GNOME-style `Super` shortcuts. Override per desktop in `settings.json`, e.g. KDE: `"platform_shortcuts": {"show_desktop": "win+d", "window_overview": "ctrl+f9"}`; XFCE: `{"show_desktop": "ctrl+alt+d"}`. Verify the shortcut works by pressing it yourself first. |

## Recognition

See [CALIBRATION.md](CALIBRATION.md). Quick checks: even front lighting, hand fully in frame, palm facing the
camera, 40-80 cm away.

## Reset

Delete (or rename) files in the config dir shown by `doctor`. Corrupt files are auto-renamed to `*.corrupt`
and defaults restored; your data is never silently deleted.
