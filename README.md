# ORBIT Gesture Studio

ORBIT turns webcam-recognized hand gestures into desktop actions. Create gesture-to-action
mappings in the desktop app, test them while disarmed, and arm them only when you are ready.

> ORBIT is purpose-built for desktop gesture control; features outside that focus are
> intentionally out of scope.

## What it does

- Tracks one hand with MediaPipe and recognizes common static poses: open palm, fist, and
  one, two, or three fingers.
- Lets you record custom static gestures and map built-in or custom gestures to actions.
- Includes experimental motion gestures: swipes, waves, circles, and user-recorded motions
  matched with dynamic time warping. Motion recognition has synthetic test coverage but has
  **not** been validated for accuracy with real users and webcams.
- Provides a live preview, mapping editor, tuning controls, diagnostics, and a headless mode.
- Runs actions through an explicit allowlist. Mappings cannot run shell commands or scripts.

Available actions include focus mode, approved keyboard shortcuts, window snapping, showing
the desktop, the OS window overview, and a log-only test action. Support depends on the
operating system and desktop environment:

| Action | Windows | Linux (X11) | macOS |
|---|---|---|---|
| Focus mode | Yes | Yes | Yes |
| Keyboard shortcuts | Yes | Yes | Yes, with Accessibility permission |
| Snap window left/right | Yes | GNOME-like desktops | Not supported by default |
| Show desktop | Yes | GNOME-like desktops | F11 by default |
| Window overview | Yes | Not supported by default | Ctrl+Up by default |

Linux Wayland sessions restrict synthetic keyboard input and global hotkeys. Use an X11
session for those actions. Shortcuts can be customized in `settings.json`.

## Install

Requirements: Python 3.10 or newer and a webcam. Python 3.11 or 3.12 is recommended for
MediaPipe wheel availability.

### Windows (PowerShell)

```powershell
git clone https://github.com/kaus-thetaa/orbit-gesture-studio.git
cd orbit-gesture-studio
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Linux or macOS

```bash
git clone https://github.com/kaus-thetaa/orbit-gesture-studio.git
cd orbit-gesture-studio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Debian or Ubuntu, `bash scripts/setup_linux.sh` can install the required system
libraries, create the environment, and install ORBIT.

## Run

```bash
python -m orbit doctor       # check the environment
python -m orbit              # open Gesture Studio
python -m orbit cameras      # find available camera indexes
python -m orbit run --preview  # headless preview; press q to quit
```

The hand model downloads from Google's MediaPipe model storage on first use. To download it
before launching the GUI, run `python -m orbit download-models`.

ORBIT starts **disarmed**. In the Live tab, confirm that the expected gesture is recognized
before using **Arm actions**. The emergency-stop button and `Ctrl+Alt+Shift+E` stop active
actions; `Ctrl+Alt+Shift+G` toggles arm/disarm. Press `Esc` in the ORBIT window to disarm.
The headless `--armed` option executes actions and should only be used deliberately.

## Mappings and data

Use the Gesture Studio tab to add, edit, enable, and remove mappings. The example mappings
are in [`config/example_mappings.json`](config/example_mappings.json). Mapping and settings
files are validated before use; invalid mappings are reported and not executed. Run
`python -m orbit check-config` to validate the local configuration.

Settings, mappings, custom gesture samples, downloaded models, and logs are stored outside
the repository in the platform's user-data directory. Set `ORBIT_HOME` to use another
directory. Camera frames are processed in memory and are not saved or sent to a service.
The one-time model download is the only network access made by the application.

## Development and limitations

Run the test suite with:

```bash
python -m pytest -q
```

Tests use synthetic landmarks and fake devices; they do not require a webcam or a downloaded
model. They do not certify gesture accuracy or desktop-action behavior on a particular
machine. Recognition depends on lighting, camera placement, hand orientation, and tuning.
Only one hand is tracked, and static poses are the primary gesture-control path.

See the [calibration guide](docs/CALIBRATION.md), [motion gesture guide](docs/MOTION_GESTURES.md),
and [troubleshooting guide](docs/TROUBLESHOOTING.md) for setup and tuning details.

## License

ORBIT Gesture Studio is distributed under the [MIT License](LICENSE).
