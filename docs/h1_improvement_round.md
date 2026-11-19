# H1 improvement round (fixed before running)

Written before any of the runs below. Train and validation games only; the test split stays
untouched until one design is frozen.

## Starting point

D0 = `configs/basketevent_h1_final.json` on cohort v5: unblocked jump shots, stronger tabular
baseline, xgboost limited to depth 2 and 50 shots per leaf, 5 date-blocked folds. Box features
change Brier by -0.0009 (95% game-bootstrap interval -0.0017 to -0.0001). Ablation showed most of
the gain comes from `release_height_bh`, so this round only tries to measure release height better.

## Steps, applied cumulatively

1. **E1, release rule.** Candidate rules for the release frame are scored against the 25
   hand-marked review clips, leaving out item 20 (the reviewer's frame 179 lies outside its strip
   and is unresolved). The rule with the most frames within 1 of the reviewer (ties: lower median
   absolute error) is carried forward only if it beats the current rule on that count.
2. **E2, smoothing.** Release height is the median over the release frame and its two neighbours.
3. **E3, player-relative release.** Add release height minus the shooter's mean release height
   over earlier-date shots in the cohort (blank with fewer than 5 such shots).

Design D1 = D0 + E1 (if carried forward), D2 = D1 + E2, D3 = D2 + E3.

## Decision rule

Each step is kept only if its box model beats the box model of the last kept design on the shots
both score: paired game bootstrap (1,000 resamples) of the Brier difference, kept when the mean
improvement exceeds one bootstrap standard error (95% interval width / 3.92). A step that is not
kept is dropped and the next step builds on the last kept design. The final design is the last
kept one; its H1 comparison (boxes vs baseline) is the reported result, and every step's result is
reported, kept or not.

## Round 1 results

E1 on the 24 hand-marked clips: a margin of 0.15 box heights above the shooter's box (was 0.25)
put 15 releases within 1 frame of the reviewer (was 10), so it was carried forward; within 3
frames fell from 88% to 79%.

| Step | Box vs baseline, Brier | 95% interval | Gain over last kept | 1 SE | Kept |
|---|---|---|---|---|---|
| D0, current final | -0.00093 | -0.00169 to -0.00008 | | | |
| D1 = D0 + E1 release rule | -0.00096 | -0.00170 to -0.00020 | 0.00000 | 0.00028 | no |
| D2 = D0 + E2 smoothing | -0.00086 | -0.00165 to -0.00005 | -0.00007 | 0.00011 | no |
| D3 = D0 + E3 player-relative | -0.00101 | -0.00181 to -0.00022 | 0.00008 | 0.00014 | no |

No step was kept; D0 stays the final design. A more accurate release frame did not change the
predictions, so the signal in release height is posture at release rather than its timing.

# Round 2 (fixed before running)

Same starting point (D0), same folds, same decision rule, steps applied cumulatively. Each adds one
posture measurement from the boxes, all at the release frame r, in the shooter's standing body
height (bh):

1. **F1, arm extension.** Shooter box height at r over standing height: how far the arms are up.
2. **F2, release vs jump peak.** Seconds from the highest point of the feet within 8 frames of r to
   r; negative when the ball leaves on the way up.
3. **F3, lean.** Sideways offset of the ball from the feet at r, signed so that positive is toward
   the basket (from the ball's sideways flight), in bh.

R1 = D0 + F1, R2 = last kept + F2, R3 = last kept + F3.

## Round 2 results

| Step | Box vs baseline, Brier | 95% interval | Gain over last kept | 1 SE | Kept |
|---|---|---|---|---|---|
| D0, current final | -0.00093 | -0.00169 to -0.00008 | | | |
| R1 = D0 + F1 arm extension | -0.00100 | -0.00182 to -0.00017 | 0.00007 | 0.00017 | no |
| R2 = D0 + F2 release vs jump peak | -0.00106 | -0.00183 to -0.00028 | 0.00014 | 0.00016 | no |
| R3 = D0 + F3 lean | -0.00078 | -0.00155 to +0.00002 | -0.00015 | 0.00018 | no |

No step was kept; D0 stays the final design. Scripts: `scripts/h1_round1_release_rules.py`,
`scripts/h1_round1.py` and `scripts/h1_round2.py` (each takes a scratch directory for the cohorts
and runs it builds); summaries in `reports/basketevent_h1_round1/` and `reports/basketevent_h1_round2/`.
The full log of H1 experiments is in `h1_experiments.md`.
