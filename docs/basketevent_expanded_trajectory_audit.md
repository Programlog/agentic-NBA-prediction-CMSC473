# Expanded BasketEvent Trajectory Audit

## Purpose

This experiment expands the release-preprocessing investigation from PR #36 to a reproducible sample of BasketEvent training and test clips.

The goal is to measure whether the existing `steady()` shooter-box filter removes frames that would otherwise satisfy the frozen release detector's geometric candidate criteria.

This analysis does not retrain models, change detection thresholds, or modify frozen H1 predictions.

## Dataset and sampling

- Source: `zaywas/BasketEvent` on Hugging Face
- Pinned dataset revision: `d28cf284f2c4cc193bf9c88ccf85524193993984`
- Random seed: `473`
- Sampling procedure: Independently select 200 JSON paths from each split using Python's `random.Random(seed).sample()` on sorted paths
- Splits analyzed: `train` and `test`
- Total sample: 400 JSON files
- Video files were not required
- Validation data was excluded from this pilot

The acquisition script downloads missing files when explicitly requested and otherwise uses existing local JSONs.

## Results

| Measurement | Train | Test |
|---|---:|---:|
| Sampled clips | 200 | 200 |
| Eligible uniquely identified shot clips | 101 | 123 |
| Clips losing at least one candidate | 10 | 6 |
| Affected percentage of eligible clips | 9.9% | 4.9% |
| Candidate frames removed | 34 | 12 |
| Clips losing every raw candidate | 0 | 0 |
| Release not found | 9 | 9 |
| Multiple candidate episodes | 63 | 83 |
| Single candidate episode | 29 | 31 |
| Ambiguous shot event | 1 | 3 |
| No identified shot event | 98 | 74 |

Across the 400 sampled files, 224 clips contained an eligible shot event. Among those, 16 clips lost at least one candidate following shooter-box preprocessing, representing 7.1% of eligible clips in the combined sample.

A total of 46 candidate frames were removed.

## Interpretation

The results extend the PR #36 finding by demonstrating candidate removal in both the training and test splits.

The original PR #36 local audit identified 10 affected clips among 97 eligible test clips and one case in which all 18 raw candidates were removed.

The present 400-file pilot found no cases of complete candidate loss. This does not contradict the earlier result because the samples differ.

Candidate removal demonstrates a change in the detector's qualifying geometry, not necessarily an incorrect physical release prediction.

## Reproduction

Run the audit from the repository root:

```powershell
.\.venv\Scripts\python.exe tools/audit_basketevent_trajectory_sample.py `
    --root data/basketevent `
    --output-dir artifacts/expanded_trajectory_audit_hashed `
    --per-split 200 `
    --splits train test
```

Add `--download` if selected JSON files are missing locally.

The program produces a sample manifest, per-clip measurements, a split-level summary, and metadata identifying the dataset revision and sampling procedure.

The manifest records SHA-256 hashes of the local files. These can be compared across runs to detect changes, but they do not independently authenticate pre-existing local data against Hugging Face.

## Limitations

- This is a 400-file pilot, not a complete evaluation of all 35,488 BasketEvent JSONs.
- The dataset contains event types other than shots; only clips meeting the existing shot-selection criteria contribute to the eligible-shot denominator.
- Training and test percentages are exploratory descriptive measurements.
- The analysis has not established a statistically significant difference between dataset splits.
- Some events may be clustered within the same games, so clip-level observations should not automatically be treated as independent.
- Raw geometric candidates are not manually verified release frames.
- Local cached files are not independently validated against the remote revision.
- The previously frozen H1 evaluation remains unchanged.
- No threshold tuning or model selection should be performed using the held-out test results.

## Conclusion

The expanded pilot confirms that the existing shooter-box preprocessing filter can suppress geometrically qualifying release candidates in both training and test clips.

The results justify further investigation and independent manual validation, but they do not demonstrate that removing or weakening the filter would improve release detection or downstream shot prediction.