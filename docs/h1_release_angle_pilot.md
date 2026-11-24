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

## Results (run once, 2026-11-24, plan at commit `c81caa4`)

7,726 development shots in 184 games; the recomputed release frame matched the cohort's on all of
them. Angle measured on 4,779 (62%): not measured because aimed toward or away from the camera
2,736, closer than 8 ft 175, no camera reference 19, ball missing 17. Output:
`reports/basketevent_h1_release_angle/` (`angles.csv`, `summary.json`, `release_angle.png`).

| Check | Measured | Result | Verdict |
|---|---|---|---|
| Gravity | 7,577 | median 11.8 m/s^2 (IQR 10.1 to 13.7) | pass |
| Plausible range | 4,779 | median 47.1 degrees (IQR 40.9 to 55.7); 95% within 20 to 75 | pass |
| Reliability (split-half) | 4,734 | rho 0.75 (0.73 to 0.77); early half 49.4, late half 44.0 degrees | pass |

The ball decelerates downward at about gravity once scaled by the shooter's box (11.8 against
9.8; a 2.0 m body is a rough scale, and a box nearer 1.7 m would match exactly), the angles centre
where published NBA release angles do (mid-40s to low 50s), and the angle drops from the first
half of the window to the second, as an arc should.

**Exploratory, not planned (checks on the reliability check):**

- Both halves are divided by the same shot-direction factor, which could make them agree on its
  own. It does not drive the result: without the direction correction the split-half rho is 0.77;
  with the direction factor's rank removed from both halves, 0.74; on shots aimed almost straight
  along the court (`cos_phi > 0.95`, 1,706 shots), 0.60.
- **The spread is too wide to be only the physical angle.** The angle varies across shots with a
  standard deviation of 12.7 degrees, roughly three times the few degrees published release
  angles vary among NBA jump shots. The split-half difference suggests about 9 degrees of noise
  per half-window angle. Part of what stays consistent between the halves is therefore likely
  viewpoint, not physics: the uncorrected depth part of the flight and camera tilt, which differ
  by shot location and arena and are the same for both halves. The angle still correlates with
  the direction factor (rho -0.25), which a true release angle should barely do.

**Reading:** all three pre-registered checks pass, so by the plan the angle is usable as an
approximate box feature. It is an image-based launch angle that behaves like a ball arc (gravity,
plausible centre, consistent within a shot), not a calibrated physical release angle, and it is
missing for 38% of shots by design. Adding it to the model would be a separate, written-down
cross-validation round, labelled exploratory; the test split is not scored again.
