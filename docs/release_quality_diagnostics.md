# Release-Candidate Quality Diagnostics

## Purpose

This diagnostic examines how the existing BasketEvent release-frame detector selects candidate frames from shooter and ball bounding-box trajectories.

The original detector chooses the last frame satisfying its geometric release criteria. Because multiple frames can satisfy those criteria, this analysis investigates whether candidate frames form one continuous sequence or several distinct sequences.

The diagnostic is read-only. It does not modify the original release detector, regenerate H1 model features, or change previously reported experimental results.

## Implementation

The implementation is in `src/nba_shots/release_diagnostics.py`, with regression tests in `tests/test_release_diagnostics.py`.

For each local BasketEvent JSON clip, the diagnostic:

1. Identifies a uniquely annotated shooter associated with a shot event.
2. Reproduces the existing release detector's geometric candidate criteria.
3. Checks that the last candidate matches the original detector's selected release frame.
4. Counts raw candidate episodes, where any missing candidate frame separates two episodes.
5. Counts gap-tolerant episodes, allowing interruptions of up to two missing frames.
6. Measures the largest frame separation between consecutive candidates.
7. Flags candidate separations of at least 25 frames for further investigation.

The gap tolerance and 25-frame screening threshold are exploratory settings, not independently validated release-detection thresholds.

The diagnostic distinguishes clips with missing shot events, ambiguous shot-event annotations, missing ball trajectories, and cases where no release candidate is found.

## Local Evaluation

The diagnostic was run on 460 locally available BasketEvent JSON files.

| Diagnostic result | Clips |
|---|---:|
| Total JSON clips scanned | 460 |
| No annotated shot event | 361 |
| Ambiguous shot event | 2 |
| Release not found | 4 |
| Clips with release candidates | 93 |
| Multiple raw candidate episodes | 71 |
| Multiple gap-tolerant candidate episodes | 61 |
| Candidate separation of at least 25 frames | 38 |

Of the 93 clips with release candidates, 71 (76.3%) contained multiple raw candidate episodes.

Allowing up to two missing frames between candidates reduced this count to 61 (65.6%).

A total of 38 clips (40.9%) contained at least one separation of 25 or more frames between consecutive candidates.

At the BasketEvent frame rate of 25 FPS, 25 frames correspond to approximately one second.

These results demonstrate that multiple geometrically plausible release candidates can occur within a single clip and that some candidates are separated by substantial intervals.

## Interpretation

Small interruptions between candidate frames may result from ordinary trajectory noise or temporary detection failures.

Larger separations may indicate competing shooting-related movements, passes, catches, or other ball movement that satisfies the detector's geometric conditions.

The existing H1 failure-case review documents two examples where the detected release frame did not correspond to the actual shooting release.

However, this diagnostic does not independently determine which candidate represents the true physical release.

Candidate ambiguity must not be interpreted as a verified detection error.

## Validation

The implementation was tested with synthetic trajectory cases covering:

- Agreement with the original release detector.
- Multiple candidate episodes.
- Short interruptions between candidates.
- Widely separated candidates.
- Invalid candidate ordering.
- Missing upward ball movement.
- Ambiguous shot-event annotations.
- Correct calculation of candidate-gap measurements.
- Missing measurement handling.
- CSV generation and overwrite prevention.

At the time of evaluation, the complete repository test suite reported **290 passed and 2 skipped**.

Ruff linting passed.

## Reproduction

From the repository root, with the project environment installed:

```powershell
.\.venv\Scripts\python.exe -m nba_shots.release_diagnostics `
    --input data/basketevent `
    --output artifacts/release_diagnostics_local.csv
```

The output file must not already exist.

The diagnostic reads locally available BasketEvent trajectory JSONs and writes a separate CSV report.

The input dataset is not modified.

## Limitations and Future Work

The 460 locally available JSON clips are not established as a representative sample of the full BasketEvent dataset.

Most clips do not have a uniquely annotated shot event, so the analysis of candidate ambiguity is restricted to clips for which the detector can produce candidates.

The diagnostic has not been evaluated against a representative set of independently verified physical release frames. Consequently, it cannot establish release-detection precision, recall, or improved release-timing accuracy.

The next meaningful research step would be to compare flagged and unflagged clips against independently reviewed release annotations, measure false-positive and false-negative rates, and assess whether additional trajectory checks improve release selection without reducing usable shot coverage.

The frozen H1 model, features, predictions, and reported results remain unchanged.