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

## Results

Script: `scripts/h1_box_error_round.py` (takes a scratch directory for the cohorts it builds).
Summary: `reports/basketevent_h1_box_error_round/`. The rebuilt D0 cohort matched v5 exactly.

**Measurement check.** G1 marked the 24 hand-marked releases exactly as the current rule does (10
within 1 frame, 21 within 3), so it was run.

| Step | Shots | Box vs baseline, Brier | 95% interval | Gain over last kept | 1 SE | Kept |
|---|---|---|---|---|---|---|
| D0, current final | 7,726 | -0.00093 | -0.00169 to -0.00008 | | | |
| G1 = D0 + release clears the head | 7,660 | -0.00040 | -0.00115 to +0.00041 | -0.00038 | 0.00020 | no |
| G2 = D0 + duplicate defender boxes | 7,726 | -0.00076 | -0.00152 to +0.00004 | -0.00017 | 0.00011 | no |
| G3 = D0 + scale check | 7,726 | **-0.00118** | -0.00199 to -0.00040 | 0.00025 | 0.00022 | yes |

What each step touched (all 15,800 cohort shots, every family and split; no outcomes):

- **G1** removed 286 shots (no release clears the head) and moved the release of 586 more. It did
  not change review items 1 and 6, the two it was written for: in both, the ball after the false
  "release" does go above the shooter's box. The model got worse on the shots it does change.
- **G2** changed the defender features of 533 shots and did not catch item 2 (that duplicate box
  overlaps the shooter by less than 0.5). Also worse.
- **G3** blanked the box features of 1,635 shots (7.4% of unblocked jump shots). It caught item 12
  (release height 2.36 from a thin stance box) but not item 4 (the far-side shooter's standing box
  is over 50 px and steady). It is kept by the rule, narrowly: the gain (0.00025) is just over one
  standard error (0.00022), and its own 95% interval (-0.00017 to +0.00070) includes zero.

**Final design of this round: D0 + G3.** Box features lower cross-validated Brier by 0.00118
(-0.00199 to -0.00040) against D0's 0.00093. Exploratory: chosen after the test split was scored,
so it cannot be confirmed on it, and the step's gain is about the size of the seed-to-seed spread
seen for D0 (section 22 of `h1_experiments.md`). The headline H1 result stays D0's.

**Reading.** Leaving shots with an implausible body-height unit blank helps a little; trying to
correct release frames or defender boxes with simple geometric rules does not. Neither G1 nor G2
fixed the specific errors they were written for, so the video-reviewed errors are not the common
case these rules caught, and the shots they did change were apparently measured well enough
before. That G1 hurt also suggests some "low releases" it rejected carried real signal.

## Seed check for G3 (fixed before running)

Added after the results above, because G3's gain (0.00025) is close to the seed-to-seed spread
seen for D0. Same cohorts, folds and decision rule; only the XGBoost seed changes.

- Run D0 and D0 + G3 cross-validation for seeds 1 to 5 (seed 42 is above).
- For each seed: G3's gain over D0 and its H1 comparison (boxes vs baseline).
- Seed-averaged: average each shot's probability over the six seeds for each design, then the same
  paired game bootstrap.
- **G3 counts as robust** if the seed-averaged gain over D0 exceeds one bootstrap standard error
  **and** the gain is positive for at least 5 of the 6 seeds. Otherwise the round's final design
  reverts to D0. Either way it stays exploratory.
