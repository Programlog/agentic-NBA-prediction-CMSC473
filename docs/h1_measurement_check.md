# H1 measurement check (fixed before running)

Written before any of the measurements below were computed. No shot outcomes are read: this
checks whether the box features measure what they claim, not whether they predict makes.

## Why

D0's gain comes mainly from `release_height_bh`. Every box feature comes from one source,
BasketEvent's detector boxes. If an independent look at the same video frames disagrees with the
boxes, the H1 result cannot be read as "release posture helps". This check comes before the
one-time test-split run.

## Population

Every cohort shot with `box_status == "ok"` and a BasketEvent video (`has_video`), from a cohort
rebuilt with `collect-basketevent` and `prepare-basketevent` under
`configs/basketevent_h1_final.json`. Results are reported for all of these shots and separately
for D0's population (unblocked jump shots). Box trajectories and videos share frame indices (checked
on 8 clips: equal frame counts, 25 fps, 1280x720), so the box release frame `r` is used as the
video frame. Each clip is downloaded, measured, and deleted; nothing is filtered on outcome.

## Measurements

All pixel values are at full 1280x720 resolution; y grows downward.

1. **Release height, YOLO pose (primary).** `yolo11x-pose.pt` on frame `r`. The shooter is the
   person whose box has the highest IoU with the BasketEvent shooter box, kept if IoU >= 0.5.
   Keypoints count when confidence >= 0.5.
   - pose release height = (mean y of the visible ankles - y of the higher visible wrist) /
     torso length (shoulder midpoint to hip midpoint), all at frame `r`.
   - It shares no input with `release_height_bh` (box feet, ball box, box standing height).
2. **Ball at the hands (secondary).** Distance from the ball box centre to the higher wrist at
   `r`, in torso lengths. Checks that `r` is a frame with the ball still near the hands.
3. **Jump, CoTracker3.** Window `r-20 .. r`, frames resized to 384 px on the long side.
   Query points at frame `r`: 5 points across the lower shooter box (x at 30-70% of width, y at
   85% of height), and a 6x4 grid of background points outside every player box. Track jump
   (px) = median y of the shooter points over frames `r-20 .. r-10` - median y at `r`, mapped
   back to full resolution. Compared with box jump (px) = the same difference of the box feet,
   which is `jump_bh` before dividing by standing height. Camera-corrected track jump subtracts
   the background points' median displacement over the same frames.
4. **Ball rise, CoTracker3.** Window `r .. r+4`, one query point at the ball box centre at `r`.
   Track rise (px) = y at `r` - y at `r+4`. Compared with box rise (px) = the same difference of
   ball box centres, which is `ball_rise_bhps` before scaling.

Tracks count only where CoTracker marks the point visible at both ends of the difference.

## Pass rules

Agreement is Spearman's rank correlation between the box value and the independent value, with a
95% bootstrap interval over shots (1,000 resamples, seed 42).

| Check | Compares | Passes when |
|---|---|---|
| Release height | `release_height_bh` vs pose release height | rho >= 0.5 and interval lower end > 0.3 |
| Ball at the hands | ball-to-wrist distance | <= 1 torso length in >= 70% of measured shots |
| Jump | box jump px vs raw track jump px | rho >= 0.5 and interval lower end > 0.3 |
| Ball rise | box rise px vs track rise px | rho >= 0.5 and interval lower end > 0.3 |

Coverage (downloaded, pose matched, keypoints visible, tracks visible) is reported for every check,
with each failure reason counted. A check with fewer than 100 measured shots is reported as
inconclusive rather than passed or failed.

## What follows

- Release height passes: D0 is read as measuring release posture; go on to the test split.
- Release height fails or is inconclusive: stop and report before any test-split run.
- Jump and ball rise: reported as feature-quality results. They do not gate the test split,
  because D0's gain does not depend on them.
- The camera-corrected jump is descriptive only; it shows how much camera motion the box jump
  carries.

## Results (pose pass, 2026-11-23)

Cohort rebuilt on Windows from BasketEvent revision `c8629460`; its D0 population is identical to
cohort v5 (same 7,726 shot IDs in 184 games, same labels). The dataset now publishes more videos
than when v5 was built, so 1,492 cohort shots have video (v5: 778). All 1,492 were downloaded and
processed; YOLO pose matched the shooter (IoU >= 0.5) on 1,196 and gave a full release height on
1,181 (no matching person 277, no wrist 19, no ankle 15). Raw rows:
`artifacts/h1_measurement_check/pose.jsonl`; summary: `.../summary.json`.

| Check | Population | Measured | Result | Verdict |
|---|---|---|---|---|
| Release height | all video shots | 1,181 | rho 0.51 (0.46 to 0.56) | pass |
| Release height | unblocked jump shots (D0) | 774 | rho 0.32 (0.25 to 0.39) | fail |
| Ball at the hands | all video shots | 1,196 | 55% within 1 torso | fail |
| Ball at the hands | unblocked jump shots (D0) | 782 | 49% within 1 torso | fail |

The plan named both populations but not which one gates. Exploratory, not planned: agreement
within each shot family is layup 0.56 (216 shots), dunk 0.53 (38), floater 0.44 (101), hook 0.33
(42), jump 0.32 (784). The pooled pass mostly reflects release height differing between shot
families; within jump shots, where D0's gain lives, box and pose release height agree only
weakly, and the detected release frame often has the ball more than a torso length from the
higher wrist.

**Decision (with the team):** read release height as "a box-derived height near release that
tracks pose only loosely within jump shots", not as a clean measure of release posture. The
test-split run tests prediction, which this check does not overturn, and goes ahead under the
protocol in `h1_test_protocol.md`, reporting this limitation alongside it.

The CoTracker3 pass (jump, ball rise) is reported below, after the amendment that fixed its sample.

## Amendment before the CoTracker3 pass (2026-11-23)

Written before any tracking measurement beyond the 3-clip mechanical pilot. CoTracker3 takes about
30 s per shot on this CPU, so all 1,492 video shots would take about 12 hours. The tracking checks
(measurements 3 and 4) therefore run on a fixed random sample:

- **Population:** D0's unblocked jump shots with video (911 shots, 22 games, all in the training
  split), the shots H1's gain comes from and where release height agreed weakly with pose.
- **Sample:** 500 of them, drawn uniformly with `numpy.random.default_rng(42)` from the pool
  sorted by shot ID, without reading outcomes or any measurement. It covers all 22 games. The IDs
  are in `reports/basketevent_h1_measurement/track_sample.txt`.
- **Unchanged:** the measurements, pass rules (rho >= 0.5 with interval lower end > 0.3), the
  100-shot minimum, and reporting of every failure. Results are reported for this sample only.

## Results (CoTracker3 pass, 2026-11-24)

All 500 sampled shots (22 games) were downloaded and tracked without errors. A value counts only
where CoTracker marks the point visible at both ends of the difference. Jump was measured on 359
shots: 140 had no shooter point visible at both ends, and 2 had no box value. Ball rise was measured
on 423: 76 had the ball not visible at both ends, and 6 had no box value (some shots had both
problems). Raw rows: `artifacts/h1_measurement_check/track.jsonl`; summary (both passes):
`reports/basketevent_h1_measurement/summary.json`.

| Check | Measured | Spearman rho (95% interval) | Verdict |
|---|---|---|---|
| Jump: box vs raw track | 359 | 0.60 (0.51 to 0.67) | pass |
| Ball rise: box vs track | 423 | 0.09 (0.00 to 0.18) | fail |
| Jump: box vs camera-corrected track (descriptive) | 359 | 0.43 (0.34 to 0.52) | not a check |

The camera's median absolute vertical motion over the jump window is 7.4 px, against a median box
jump of 12.5 px. Part of what both the box jump and the raw track jump pick up is the camera moving,
which is why agreement drops once camera motion is removed.

Exploratory, not planned: the ball-rise failure looks like the tracker, not the boxes. Over the 4
frames after release the box ball centre rises a median 66.5 px, but the tracked ball point moves a
median -5.5 px (|9.3| px), so it mostly stays on the hands or background instead of following the
ball. This matches the 25-clip pilot (`h1_experiments.md` section 17), where CoTracker3 kept losing
the ball. The pre-registered verdict stays fail; this check cannot say whether box ball rise is
accurate.

**Reading:** box jump is supported as a jump measurement, with camera motion mixed in. Box ball
rise is unverified. Neither gates the test split, and D0's gain comes mainly from release height
(section 14 of the log), so the H1 result and its release-height caveat are unchanged.
