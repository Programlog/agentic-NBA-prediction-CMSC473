# H1 test-split protocol (fixed before running)

Written before any test-split shot was scored. This is the one test-split evaluation of H1; it is
not repeated or adjusted after its results are seen.

## Design (frozen)

D0 = `configs/basketevent_h1_final.json` on the rebuilt cohort
(`data/processed/basketevent_cohort_rebuild`, identical in shot IDs and labels to cohort v5):
unblocked jump shots; baseline features shot distance, game clock, period, shot value, court x/y,
home/away, shot type, zone, area, and player history; XGBoost depth 2, 50 shots per leaf, 200
trees. The box model adds the 9 box features.

## Fit and score

- Models are fit on training games only (152 games, 2025-10-22 to 2025-11-10) with
  `nba-shots train`, then scored once on the test games (34 games, 2025-11-16 to 2025-11-20)
  with `nba-shots evaluate --split test --final`. Validation games are not refit into training,
  matching how every earlier validation result was produced.
- Six XGBoost seeds: 42, 1, 2, 3, 4, 5. On this machine D0's cross-validated effect ranged from
  -0.00073 (seed 42, not significant) to -0.00098 across these seeds, so a single seed could land
  on either side of significance by chance.

## Primary result

For each of `xgboost` (baseline) and `xgboost_boxes`, average each test shot's predicted
probability over the six seeds. H1 is the Brier difference, box model minus baseline, on all test
shots, with a 95% paired bootstrap interval over whole games (1,000 resamples, seed 42), via
`evaluation.paired_comparison`. Negative favours the boxes; supported if the interval lies
entirely below zero.

## Also reported

- The same comparison for each seed separately, seed 42 first.
- Brier, log loss and ROC-AUC of constant, logistic regression, baseline and box models.
- The same averaged comparison on the validation games from the same fits, for reference.
- The release-height limitation from `h1_measurement_check.md`.

## Output

`reports/basketevent_h1_test/`: the seed-averaged predictions, `summary.json`, and a results
section appended to this file. Per-seed runs and fitted models stay in `artifacts/h1_test_run/`.
(The seed-averaged prediction CSVs were later kept off `main` for size; they are in commit
`8955c98` on `feature/h1-integrated-cotracker`.)

## Results (run once, 2026-11-23, code at commit `15f6eea`)

H1 primary, six seeds averaged, 1,546 test shots in 34 games: Brier **-0.00067** (95% interval
-0.00237 to +0.00096). Not supported at 95% on the test split alone; same direction and size as
the cross-validated result (-0.0009), with every seed negative (-0.00029 to -0.00088). Baseline
Brier 0.2010, box model 0.2003; AUC 0.605 to 0.609. Full tables in `h1_experiments.md` section 24.
