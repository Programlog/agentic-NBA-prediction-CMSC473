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

The CoTracker3 pass (jump, ball rise) has not been run.
