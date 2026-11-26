
# Release Candidate Loss from Shooter-Box Preprocessing

## Purpose

This exploratory audit measures whether the existing `steady()` shooter-box
filter removes frames that would otherwise satisfy the frozen H1 release
detector's geometric candidate rules.

It does not modify the frozen detector, retrain a model, or change any
previously reported H1 results.

## Method

The diagnostic compares candidate frames computed from two versions of
the same annotated shooter's tracking trajectory:

1. Raw valid bounding boxes, before the `steady()` filter.
2. Filtered bounding boxes, after the original `steady()` function.

Both use the existing `release_candidates()` criteria.

The difference identifies candidate frames removed by preprocessing.
A raw-geometry candidate is not independently verified as a true
physical shot release.

## Data scope

- Input: `data/basketevent`
- Total local JSON files scanned: 460
- Eligible annotated shot clips: 97
- Eligible clips by split: `test` = 97
- Excluded from this denominator: 361 clips without an identified shot event
  and 2 with ambiguous shot events
- The diagnostic analyzed existing local files, not the complete
  Hugging Face BasketEvent dataset.

## Results

| Measurement | Result |
|---|---:|
| Eligible annotated shot clips | 97 |
| Clips losing at least one raw candidate | 10 |
| Proportion of eligible clips affected | 10.3% |
| Total raw candidate frames removed | 37 |
| Clips losing every raw candidate | 1 |
| Clips with no selected release after filtering | 4 |

### Complete candidate-loss example

Clip:

`test/0022500003/ce979297-2c6d-a6df-2455-eac10a268ed4_1280x720.json`

The annotated event is a missed running reverse layup by Bridges.

- Raw-geometry candidate frames: 18
- Candidates remaining after `steady()`: 0
- Removed candidates: 18
- Frozen release detector result: `release_not_found`

Inspection of the video suggests a shot release around frames 32–34,
but the exact physical release frame remains uncertain.

The raw shooter bounding boxes exist near this event, while the
filtered trajectory contains missing boxes. This is consistent with
the preprocessing filter suppressing the required candidate geometry.

## Interpretation

The audit demonstrates that shooter-box filtering can suppress
release candidates that otherwise satisfy the original geometric
rules. In one observed annotated shot clip, every raw candidate
was removed.

This establishes a computational source of candidate loss, not
a validated improvement from disabling the filter.

The filter can also protect against tracking artifacts, so retaining
raw candidates could introduce false release detections.

## Limitations

- The local scan covers 460 JSON files, not the full BasketEvent dataset.
- All 97 eligible annotated shot clips came from the local test split.
- The selection was not a representative random sample of BasketEvent.
- Candidate-frame counts are not counts of independent shots.
- Manual video verification remains incomplete.
- This analysis is exploratory and should not be used to retune the
  frozen H1 detector on its test data.

## Reproduction

Run:

```powershell
.\.venv\Scripts\python.exe -m nba_shots.release_diagnostics `
    --input data/basketevent `
    --output artifacts/release_preprocessing_audit_all.csv
```

The resulting CSV includes:

- `raw_geometry_candidate_count`
- `candidate_count`
- `candidates_removed_by_steady`
- `removed_candidate_frames`

The output file must not already exist because the diagnostic
intentionally refuses to overwrite existing reports.