# H1 box-error round (fixed before running)

Written before any of the runs below. Exploratory: the test split was scored once
(`h1_test_protocol.md`), so nothing here can be confirmed on it, and nothing here reads it. Every
run is 5-fold date-blocked cross-validation on the 184 train and validation games.

## Why

The failure-case review (`h1_failure_cases.md`, part B) found that 4 of 6 surprising makes were box
errors that made the box model's prediction worse: two release frames at moments with no shot (a
catch, and a moment after the shot had already happened), one defender box that duplicates the
shooter, and one likely scale error. One surprising miss had an inflated release height from a thin
partial stance box. This round tries one fix per error type.

## Steps, applied cumulatively

Starting point D0 = `configs/basketevent_h1_final.json` on a cohort rebuilt from
`data/basketevent_labels_v2` (must reproduce v5's shots and features exactly).

1. **G1, release must clear the head.** A candidate release frame k is accepted only if, within the
   7 frames after k, the ball's centre rises above the top of the shooter's box at k (the raised
   hands, or the head). A catch, pass or dribble starts at the waist or chest and does not clear
   the head; a shot does. If no candidate passes, the shot has no release and leaves the cohort, as
   now.
2. **G2, duplicate defender boxes.** An opponent whose box at the release frame overlaps the
   shooter's box with intersection over union of 0.5 or more is treated as a second detection of
   the shooter and ignored for every defender feature.
3. **G3, scale check.** All nine box features are left blank (the shot stays in the cohort) when the
   standing height used as the unit is under 50 pixels, or differs by more than 25% from the median
   height of the shooter's box over all frames before release.

G1, G2 and G3 thresholds are fixed here and not tuned.

## Measurement checks (no outcomes; can only stop a step)

- G1 on the 24 hand-marked review clips (item 20 left out, as in round 1): if it puts fewer
  releases within 1 frame of the reviewer than the current rule, G1 is not run.
- For each step: the share of cohort shots whose features change, and whether it changes the
  failure-review shots it was written for (G1: items 1 and 6; G2: item 2; G3: items 4 and 12).
  Reported, not used to decide.

## Decision rule (same as rounds 1 and 2)

Each step is kept only if its box model beats the box model of the last kept design on the shots
both score: paired game bootstrap (1,000 resamples) of the Brier difference, kept when the mean
improvement exceeds one bootstrap standard error (95% interval width / 3.92). A step that is not
kept is dropped and the next step builds on the last kept design. Every step's H1 comparison
(boxes vs baseline) is reported, kept or not. The headline H1 result stays D0's, cross-validated
and test-scored; a kept step is reported as an exploratory improvement.
