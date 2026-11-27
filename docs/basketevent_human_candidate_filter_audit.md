# BasketEvent Human-Referenced Candidate-Filter Audit

## Purpose

This exploratory analysis examines whether the frozen release detector's `steady()` shooter-box filter removes geometric release candidates near manually estimated shot-release intervals.

It extends the earlier 400-JSON trajectory audit with human observations from 20 BasketEvent test videos.

## Method

- Sample: 20 previously selected and manually reviewed test clips.
- Human review: 19 provisional release intervals and one uncertain release.
- Event identities: 20 user-confirmed shot-event matches.
- Inputs: corresponding local BasketEvent trajectory JSON files.
- Diagnostics: the existing `diagnose_clip()` implementation.
- Raw candidates: candidate frames before `steady()` filtering.
- Retained candidates: candidate frames after filtering.
- Removed candidates: raw candidates absent from the retained set.

An interval overlap occurs when a candidate frame falls inside the inclusive provisional human interval. Nearest-candidate distance is zero for overlapping candidates.

No detector logic, thresholds, frozen predictions, or trajectory files were modified.

## Results

| Measurement | Result |
|---|---:|
| Reviewed clips | 20 |
| Provisional release intervals | 19 |
| Uncertain release intervals | 1 |
| User-confirmed event identities | 20 |
| Multiple candidate episodes | 8 |
| Single candidate episode | 8 |
| Release not found | 4 |
| Raw candidate overlap with provisional interval | 12/19 |
| Retained candidate overlap with provisional interval | 11/19 |
| Removed candidate overlap with provisional interval | 1/19 |
| Clips losing candidates to filtering | 2/20 |
| Candidate frames removed | 21 |
| Clips where removed candidates were closer than retained candidates | 1 |

### Case 1: Phoenix Allen

Video prefix: `06287e20`

- Provisional human interval: frames 82–83.
- Raw geometric candidates: 24.
- Candidates retained: 21.
- Candidates removed: 3.
- Nearest retained candidate: 0 frames from the interval.
- Nearest removed candidate: 147 frames from the interval.

Filtering removed three candidate frames, but they were far from the provisional release. Retained candidates already overlapped the interval.

### Case 2: New York Bridges

Video prefix: `ce979297`

- Provisional human interval: frames 32–34.
- Raw geometric candidates: 18.
- Candidates retained: 0.
- Candidates removed: 18.
- Nearest removed candidate: 0 frames from the interval.
- Frozen detector status: `release_not_found`.

All 18 raw candidates were eliminated by `steady()`, including at least one candidate inside the provisional human release interval.

This demonstrates complete loss of geometrically plausible candidates in one observed clip. It does not establish that those candidates were physically correct release frames.

### Cases without raw candidates near the human interval

Seven of the 19 provisional intervals had no overlapping raw geometric candidate. These cases cannot be explained solely by candidate removal during `steady()` preprocessing.

## Special Cases and Interpretation

- **Champagnie:** The event identity was user-confirmed, but the release interval remains uncertain because multiple shooting motions appear. This clip was excluded from interval-overlap calculations.
- **Dunn:** The alley-oop attempt has ambiguous ball-hand separation despite its provisional interval.
- **Nance Jr.:** The blocked layup may represent defender-induced separation rather than normal shooter-initiated release.

These observations should not be treated as verified release ground truth.

## Reproduction

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m tools.audit_release_candidates_against_human `
    --reviews artifacts/release_manual_review/blind_review.csv `
    --json-root data/basketevent `
    --output artifacts/release_candidate_filter_audit/per_clip.csv
```

The local human-review CSV and matching BasketEvent trajectory files are required to regenerate the per-clip audit. They may not be included in version control.

## Limitations

- The sample consists of only 20 BasketEvent test clips.
- Release intervals are provisional rather than verified ground truth.
- Some frame interpretations and note wording were AI-assisted.
- Event identities reflect the reviewing user's judgments, not independent adjudication.
- Geometric candidate overlap is not proof of correct physical release detection.
- Tracking errors, occlusion, and unusual shot mechanics may affect results.
- The sample is too small to establish population-wide conclusions.
- This analysis does not establish improved detection accuracy or downstream H1 performance gains.
- Held-out test observations must not be used to tune the frozen release detector or retrospectively select model parameters.

## Conclusion

In the 20 reviewed clips, `steady()` removed candidate frames from two clips, totaling 21 frames.

The clearest example was Bridges, where all 18 raw candidates were removed, including candidates overlapping the provisional human release interval.

However, seven provisional intervals lacked overlapping raw candidates even before filtering. Therefore, preprocessing removal is only one possible source of release-detection limitations.

These findings justify further independent validation, not a claim that removing the filter improves detector accuracy.