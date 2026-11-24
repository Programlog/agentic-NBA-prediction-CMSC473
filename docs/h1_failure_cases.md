# H1 failure-case analysis (definitions fixed before looking)

Written before any per-shot error was examined. The proposal promises an analysis of "shots the
model got completely wrong". This is exploratory and descriptive: it reads predictions that
already exist, refits nothing, scores nothing new, and changes no H1 result.

## Predictions used

- **Part A, test split:** the six-seed averaged predictions of the one test run
  (`reports/basketevent_h1_test/test_predictions_seed_mean.csv`, 1,546 shots, 34 games), for the
  box model (`xgboost_boxes`) and the baseline (`xgboost`).
- **Part B, video review:** no test shot has video (all video shots are in training games), so the
  visual review uses the out-of-fold cross-validation predictions of D0 on the rebuilt cohort
  (`artifacts/h1_rebuild_cv_d0*`, seeds 42 and 1-5, averaged per shot). Each of those shots was
  predicted by a model that did not see its game.

## Definitions

- **Surprising make:** a made shot in the lowest fifth of the box model's predicted probability.
- **Surprising miss:** a missed shot in the highest fifth.
- Fifths are taken over all shots of the split being analysed. Each surprising group is compared
  with the shots in the same fifth that went the expected way (misses in the lowest fifth, makes in
  the highest), so the question is what separates a shot the model was most confident about from
  one where that confidence was wrong.
- **Changed by the boxes:** the 10% of shots with the largest |box - baseline| prediction. For
  them: how often the box model moved toward the outcome, and their Brier difference.

## Part A: test split

1. **Calibration:** predicted vs observed make rate by predicted decile, both models.
2. **Surprising makes and misses:** counts, and a profile against the same-fifth expected shots:
   shot distance, share of threes, release height, share below 0.9 body heights (the group from
   `h1_experiments.md` section 25), defender distance, share with any box feature missing,
   shooter's prior make rate, shot type.
3. **Changed by the boxes:** as defined above, with the same profile.
4. **Slices:** Brier difference (box - baseline) with 95% game-bootstrap intervals (1,000
   resamples, seed 42) for: 2 vs 3; distance under 10 ft, 10-16, 16 ft to the 3-point line, 3s;
   release height below vs at or above 0.9; any box feature missing vs none; shot type (Jump Shot,
   Pullup, Step Back, other). Exploratory with no multiplicity correction; a slice interval that
   excludes zero is reported as a pattern to check, not a finding.

## Part B: video review

Among D0 unblocked jump shots with video, the 6 most confident surprising makes (lowest
cross-validated box-model probability among makes) and the 6 most confident surprising misses
(highest among misses). For each, frames at release `r - 10`, `r` and `r + 4` with the shooter,
nearest defender and ball boxes drawn, and a one-line description of what is visible: whether the
detected release frame is right, the contest, and anything the features miss. Descriptive only.

Script: `scripts/h1_failure_cases.py`. Output: `reports/basketevent_h1_failure_cases/`.
