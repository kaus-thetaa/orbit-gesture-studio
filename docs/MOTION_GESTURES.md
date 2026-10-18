# Motion gestures

Static poses ("two fingers up") are recognised frame by frame. **Motion gestures** are recognised from the
*shape of a movement over about a second*: swipes, a wave, circles, or anything you record yourself.

## What ships

| Gesture | How it's recognised | Needs recording? |
|---|---|---|
| `swipe_left` `swipe_right` `swipe_up` `swipe_down` | long, straight, one dominant axis, quick | no |
| `wave` | ≥3 left-right direction changes, little vertical travel | no |
| `circle_cw` `circle_ccw` | ≥300° of turning, nearly closed, roughly round | no |
| your own (`zigzag`, ...) | Dynamic Time Warping against your examples | yes, 20-30 examples |

Directions are as you see them in the mirrored preview (your own left/right). `circle_cw` is clockwise on screen.
Motions are momentary: map them with mode **trigger** (fire once) or **toggle**; `hold` is rejected.

## How it works

1. **Palm tracking.** The palm centre (average of 5 landmarks) is smoothed and its speed measured in
   *hand-lengths per second*, so distance from the camera doesn't matter.
2. **Segmentation.** A movement starts when speed exceeds `start_speed` and ends after `gap_ms` of being still
   (or the hand leaving the frame). Movements shorter than `min_duration_ms`, shorter than `min_path`, or longer
   than `max_duration_ms` are discarded. There is no "start recording" gesture: it's continuous.
3. **Normalisation.** The path is expressed in hand-lengths relative to its start, so it doesn't depend on where in
   the frame you move or how big your hand looks.
4. **Classification.**
   * *Custom motions* first: the path is resampled to 32 points and compared with every example (plus 4
     automatically generated variants of each: ±15% size, ±10° rotation, different speed profile, jitter) using
     DTW, which tolerates differences in speed. The threshold comes from how much your own examples differ from each
     other (leave-one-out), scaled by `custom_threshold_scale` and the per-motion sensitivity.
   * *Built-ins* otherwise, using the rules above.
5. **Gating.** By default the hand must be an **open palm** while moving (`motion.require_pose`; `""` = any pose),
   so typing, reaching for a cup, or a fist swing don't trigger. Recorded motions require the pose you were
   holding while recording (you can untick this in the recorder).
6. **Cooldown.** `cooldown_ms` (0.8 s) after a recognised motion suppresses the *return stroke*.

## Recording a custom motion

Gesture Studio → **Record motion gesture...**

1. Type a name. Hold the pose you'll use. Make the motion, then **pause ~0.3 s**. Each completed movement becomes
   one example (watch the yellow trail in the preview).
2. Get **20-30 examples**; the minimum is 8. Vary speed, size and position slightly between them. More *varied*
   examples beat more *identical* ones.
3. Delete any that look different (icons: white dot = start, yellow = end; red = flagged as unlike the others).
4. Switch to **Test**: repeat the motion (should say MATCH), then try other movements (should say no match).
5. Save, then assign an action (trigger or toggle).

Why many samples: DTW matching has no training step, so a handful of examples works, but the acceptance
threshold is estimated from the spread of your examples; with <10 it is unreliable. The 4x augmentation helps with
the *matching*, not with an unrepresentative set. If you find 20-30 isn't enough for a complex shape, the next step
is a small sequence model trained on landmark trajectories, which would need hundreds of examples per class.

## Tuning (Tuning tab; `motion.*` in settings.json)

| Problem | Try |
|---|---|
| Nothing is detected | lower `start_speed` (1.5 → 1.0), `min_path`, `swipe_min_distance`; check `require_pose` |
| Random movements trigger | raise `start_speed`, `min_confidence`, `swipe_straightness`; keep `require_pose` on |
| Swipe followed by an unwanted opposite swipe | raise `cooldown_ms` (800 → 1200) or pause before returning your hand |
| Two quick strokes are read as one movement | strokes closer than ~0.25 s merge: lower `gap_ms` (150 → 100) or pause longer |
| Custom motion never matches | check Test mode distance vs threshold; raise `custom_threshold_scale` or record more varied examples |
| Custom motion matches too easily | lower `custom_threshold_scale`; add confusing movements as examples of *other* gestures |

The Live tab's **Last motion** line and `python -m orbit run` print why a movement was ignored, with the measured
path length, straightness, reversals and turning, so you can see which threshold to move.

## Limits (honest list)

* Defaults were validated on synthetic trajectories and automated tests; they have **not** been measured on real
  hands. Expect to tune once.
* ~0.25 s of segmentation latency after the movement ends (smoothing + `gap_ms`).
* The return stroke can still sneak through if you move back slowly and late.
* Motion is tracked at the palm centre in 2D; depth movements (pushing toward the screen) aren't captured.
* One hand, one movement at a time. Very slow movements (below `start_speed`) are intentionally invisible.
