# H1 release-angle pilot (fixed before running)

Written before any angle was computed. Exploratory and after the test run: it asks whether a
release angle can be measured from BasketEvent's boxes at all. It reads no outcomes and no test
games, adds nothing to D0, and its result is a measurement finding either way.

## Why

A true physical release angle needs a calibrated 3D camera, which broadcast clips do not give
(`PRD.md`, out of scope). An approximate angle from the ball's flight just after release can be
computed from the boxes for every shot. Before such an angle is used anywhere, it has to pass
checks that need no ground truth and no outcomes.

## Population

D0's development shots: unblocked jump shots with `box_status == "ok"` in the train and
validation games of `data/processed/basketevent_cohort_rebuild` (7,726 shots, 184 games). Box
trajectories come from the collection's per-game labels at revision `c8629460`; the shooter, the
glitch filter (`steady`) and the release frame `r` are exactly those of `basketevent.py`. Test
games and `y_true` are not read.

## Measurement

All positions are ball-box centres in full-resolution pixels; y grows downward.

1. **Camera correction.** For each frame `r + k`, the camera shift is the median displacement,
   since `r`, of the feet of every other player in the clip (glitch-filtered boxes, at least 3
   players visible at both frames). It is subtracted from the ball's displacement. Feet stay on the
   floor, so this removes pans and tilts, not zoom.
2. **Shot direction.** From the NBA shot location (`loc_x`, `loc_y`, basket at the origin, `loc_y`
   along the court's length): `cos_phi = |loc_y| / hypot(loc_x, loc_y)`, the share of the shot's
   direction that runs along the court's length, which the main broadcast camera sees side-on.
   Shots with `cos_phi < 0.7` (aimed more than about 45 degrees toward or away from the camera) or
   closer than 8 ft are not measured.
3. **Release angle (the candidate feature).** Over `r .. r+4` (the same window as
   `ball_rise_bhps`): rise = -(corrected dy), run = |corrected dx| / cos_phi,
   `angle = atan2(rise, run)` in degrees. Pixel ratios only, so no scale is needed. Camera tilt and
   the depth part of the flight are not corrected; this is an image approximation, not a physical
   angle.

## Checks (no outcomes)

| Check | What it tests | Passes when |
|---|---|---|
| Gravity | Fit y = a t^2 + b t + c to the corrected ball centre over `r .. r+12` (at least 8 frames). g = 2a x 25^2 x (2.0 m / standing box height px). Validation only, never a feature. | median g within 6.9 to 12.7 m/s^2 (9.8 +/- 30%) |
| Plausible range | Distribution of the angle | median within 40 to 60 degrees, and >= 80% of angles within 20 to 75 degrees |
| Reliability | Split-half: angle over `r .. r+2` vs over `r+2 .. r+4`, Spearman across shots, 95% bootstrap interval (1,000 resamples, seed 42) | rho >= 0.5 and interval lower end > 0.3 |

Gravity assumes a standing box of about 2.0 m (NBA players average about 1.98 m; boxes add a
margin), so it tests scale and direction only roughly; the range is wide for that reason.

Coverage is reported with each reason a shot is not measured (no camera reference, ball missing,
direction filter, too close, too few frames for the fit). A check with fewer than 100 measured
shots is inconclusive.

## What follows

- All three pass: the angle is usable as an approximate box feature. Adding it to the model would
  be a separate cross-validation round with its own written plan, labelled exploratory; the test
  split is not scored again.
- Any check fails: report it as a measurement result ("release angle is not measurable from
  broadcast boxes in this way") and stop.

Script: `scripts/h1_release_angle_pilot.py`. Output: `reports/basketevent_h1_release_angle/`.
