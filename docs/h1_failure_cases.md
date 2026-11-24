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

## Results (2026-11-24, plan at commit `44e85d9`)

Output: `reports/basketevent_h1_failure_cases/` (`summary.json`, `review.csv`, `review/*.jpg`).

### Part A: test split (1,546 shots)

**Calibration.** The box model's predicted and observed make rates agree within a few points in
deciles 1 to 9 (decile 1: 17.6% predicted, 15.5% made). The top decile is under-predicted: 42.2%
predicted, 51.0% made. The baseline is more compressed (decile 1 at 21.3%, decile 10 at 40.4%).

**Surprising makes and misses.** "Confident" is modest here: the lowest fifth averages a 19%
predicted chance and the highest 38 to 40%.

| Group | Shots | Distance (ft) | Threes | Release height (bh) | Below 0.9 bh | Defender (bh) | Box missing |
|---|---|---|---|---|---|---|---|
| Surprising makes (lowest fifth, made) | 65 | 25.6 | 98% | 1.43 | 11% | 0.77 | 6% |
| Expected misses (lowest fifth, missed) | 245 | 25.5 | 98% | 1.37 | 16% | 0.80 | 4% |
| Surprising misses (highest fifth, missed) | 176 | 14.5 | 23% | 1.66 | 1% | 0.81 | 3% |
| Expected makes (highest fifth, made) | 133 | 12.6 | 12% | 1.60 | 2% | 0.87 | 1% |

Within each fifth, the shots the model got wrong look almost the same on every feature as the ones
it got right. The lowest fifth is long threes; the highest is short and mid-range twos, and even
there misses (176) outnumber makes (133). Nothing in the features separates a surprising result
from an expected one: at this feature resolution the remaining error is mostly shot-to-shot
chance.

**Where the boxes changed the prediction most** (top 10% of |box - baseline|, 155 shots, mean
change 8.9 points): the box model moved toward the outcome on 52% of them, but its Brier there is
lower by 0.0089, and these 155 shots account for more than the whole test-split gain (155 x 0.0089
is about 1.4 against 1,546 x 0.00067, about 1.0). They are enriched for low releases (28% below
0.9 body heights vs 4.5% overall) and open shots (defender 1.25 bh vs 0.96).

**Slices** (Brier box - baseline, 95% game-bootstrap interval; none excludes zero):

| Slice | Shots | Brier difference | Interval |
|---|---|---|---|
| Release height below 0.9 bh | 69 | -0.0081 | -0.0247 to +0.0072 |
| Under 10 ft | 87 | -0.0052 | -0.0146 to +0.0042 |
| Other shot types (turnaround, fadeaway, ...) | 220 | -0.0038 | -0.0094 to +0.0017 |
| 10-16 ft | 180 | -0.0014 | -0.0078 to +0.0049 |
| 2-pointers | 400 | -0.0008 | -0.0043 to +0.0028 |
| All box features present | 1,509 | -0.0008 | -0.0026 to +0.0010 |
| 3-pointers | 1,146 | -0.0006 | -0.0025 to +0.0011 |
| Jump Shot | 830 | -0.0005 | -0.0025 to +0.0014 |
| Release height 0.9 bh or higher | 1,477 | -0.0003 | -0.0020 to +0.0012 |
| Pullup | 333 | -0.0003 | -0.0036 to +0.0030 |
| Step Back | 163 | +0.0018 | -0.0041 to +0.0077 |
| 16 ft to the 3-point line | 133 | +0.0029 | -0.0031 to +0.0084 |
| Some box feature missing | 37 | +0.0037 | -0.0145 to +0.0190 |

The point estimates agree with section 25 of the log (the gain sits in low releases), but every
slice is too small to show more than direction.

### Part B: video review (12 cross-validated training-game shots)

| # | Kind | Box model | Baseline | What we see |
|---|---|---|---|---|
| 1 | surprising make | 13% | 25% | wrong release frame: the shot clock reset before `r`, so the shot had already happened |
| 2 | surprising make | 16% | 20% | defender box is a duplicate of the shooter; the shot looks open |
| 3 | surprising make | 17% | 18% | unclear: tight closeout or duplicate box |
| 4 | surprising make | 17% | 25% | likely box scale error: small far-side shooter, ball above the head yet 0.86 bh |
| 5 | surprising make | 17% | 19% | features right: a genuinely contested three that went in |
| 6 | surprising make | 17% | 27% | wrong release frame: a catch or pass, not a shot |
| 7 | surprising miss | 57% | 40% | contest under-measured: defender on the shooter, distance 1.01 bh |
| 8 | surprising miss | 51% | 45% | shooter box merged with another player in traffic |
| 9 | surprising miss | 50% | 45% | features right: contested step-back |
| 10 | surprising miss | 50% | 46% | features right: contested short turnaround |
| 11 | surprising miss | 49% | 53% | features right: turnaround fadeaway, hand in the face |
| 12 | surprising miss | 48% | 46% | likely box scale error: thin stance box inflates release height (2.36 bh) |

- **Surprising makes are mostly measurement errors.** In 4 of 6 the boxes measured something
  else: two wrong release frames (in both, the detected "release" is a moment with no shot, which
  gives the low release height the model penalises), one duplicate defender box, and one likely
  scale error. Only one is a correctly measured contested make. In all six the box model was more
  pessimistic than the baseline, so these are cases where the box features hurt.
- **Surprising misses are mostly correctly measured contested shots.** 3 of 6 are turnarounds,
  fadeaways or step-backs with a defender's hand up that the features captured, and the model
  still gave about 50%: short distance and a high release outweighed the contest. Two more are box
  problems (merged box in traffic, inflated release height) and one under-measures a defender in
  contact.

**Reading.** On the test split the model's errors are not explained by any feature it has: the
surprising shots look like the expected ones. The video points to two concrete failure sources:
(1) box measurement errors, especially a wrong release frame, which produce exactly the low
releases the model has learned to distrust, and (2) close contests on short shots that a distance
in body heights does not express (a hand in the face at 0.5 bh reads the same as a defender
standing beside the shooter). Twelve shots are an illustration, not a rate.
