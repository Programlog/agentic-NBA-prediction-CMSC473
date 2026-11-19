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
