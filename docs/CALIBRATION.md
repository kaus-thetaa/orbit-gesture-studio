# Calibrating ORBIT

The defaults come from geometry on synthetic hands, **not** from real-user data, so expect to tune once.
Everything below is in the **Tuning** tab (applies instantly; click *Save settings* to keep it).

## 5-minute procedure

Sit at your normal distance with decent front lighting (a window or lamp in front of you, not behind). Keep the
ORBIT **Live** tab open and ORBIT **disarmed**.

1. **Distance/size.** Hold an open palm at your usual distance. *Hand* should say `tracking`. If it says
   `hand_too_small`, move closer or lower `min_hand_size`. `partial_hand` means part of the hand left the frame.
2. **Two fingers.** Raise index + middle, curl ring + pinky. *Raw gesture* should read `two_fingers` with a
   confidence ≳ 0.8. Wiggle slightly; it should stay stable.
   * Reads `open_palm` or `three_fingers` when you curl fingers loosely → raise `extension_ratio`
     (e.g. 1.35 → 1.45) or `straight_angle_deg` (+5).
   * Reads `fist` or `one_finger` when you extend fingers → lower the same values.
3. **Open palm** and **fist**. Each should read correctly. A relaxed fist that shows as `unrecognized` means your
   fingers are only half-curled - curl harder or lower `extension_ratio`.
4. **Stability.** Hold each pose for 10 s. *Confirmed gesture* must not flicker. If it drops out:
   raise `release_frames` (try 8-10) or lower `min_gesture_confidence` (0.60).
5. **Responsiveness.** If activation feels slow raise nothing - lower `confirm_frames` (6 → 4). At 30 fps,
   6 frames ≈ 0.2 s. If random poses trigger it, raise `confirm_frames` (8-10).
6. **False positives.** Type, scratch your nose, reach for a drink for a minute with ORBIT disarmed and watch the
   event list. Any unintended `PRESSED` → raise `confirm_frames` or `min_gesture_confidence`.
7. Click **Save settings**. Only then **Arm**.

## What each threshold means

| Setting | Meaning | Raise it to... |
|---|---|---|
| `confirm_frames` | consecutive matching frames before activation | reduce false triggers (slower) |
| `release_frames` | consecutive non-matching frames before release | survive tracking dropouts (laggier release) |
| `min_gesture_confidence` | per-frame minimum confidence | demand cleaner poses |
| `extension_ratio` | tip-to-wrist ÷ knuckle-to-wrist for "extended" | require straighter, longer fingers |
| `straight_angle_deg` | minimum PIP joint angle for "extended" | same, via bend angle |
| `thumb_ratio` | thumb-out threshold (only used by custom/pattern features) | - |
| `min_hand_size` | reject hands smaller than this fraction of the frame diagonal | ignore far-away hands |
| `custom_distance_scale` | custom-gesture acceptance radius multiplier | accept looser matches |

## Known limits

* Confidence is a *margin* from the thresholds, not a probability.
* Poor light, motion blur, gloves, and fingers pointing at the camera all reduce accuracy.
* Thresholds that work for you may not work for another hand size; this is why they're configurable.

## Custom gestures

Vary position/tilt/distance a little across ~20 samples, delete red (suspicious) samples, then check the **Live
match** readout before saving. If it matches too easily, lower that gesture's acceptance by lowering
`custom_distance_scale`; if it never matches, raise it or re-record with more variety.
