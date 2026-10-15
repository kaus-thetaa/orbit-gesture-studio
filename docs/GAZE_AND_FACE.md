# Gaze and face signals (EXPERIMENTAL)

Off by default. They run in a separate module and cannot affect the hand pipeline if they fail.

## Gaze

Estimates where your irises sit between your eye corners (0 = left, 1 = right). It is an *eye-in-head* signal: not
a screen coordinate, and it shifts when you turn your head.

1. `python -m orbit download-models --face`
2. `python -m orbit calibrate-gaze` - look at left edge, centre, right edge when prompted (head still).
3. In `settings.json` set `"gaze": {"enabled": true, ...}` and restart.
4. In Gesture Studio, add a mapping with gesture `gaze_left` / `gaze_right`, mode `trigger`
   (e.g. `snap_window_left`).

Safeguards: smoothing, a dwell time (`dwell_ms`, default 900 ms of sustained look), release delay and a cooldown
(`cooldown_ms`, default 1.5 s) so normal reading/scanning does not fire. Calibration is rejected if movement is
too small to separate the three zones.

Limits: sensitive to head turns, glasses glare, low light and camera height. Not tested on real hardware yet.

## Face signals

User-defined thresholds on MediaPipe blendshape scores. They are *muscle-movement measurements*, not emotions.

```json
"face": {"enabled": true, "signals": [
  {"name": "jaw_open", "blendshape": "jawOpen", "threshold": 0.6, "dwell_ms": 400}
]}
```

Then map gesture `face_jaw_open` to an action. Invalid signal definitions are skipped with a log warning.
Gaze/face settings take effect after restarting ORBIT.
