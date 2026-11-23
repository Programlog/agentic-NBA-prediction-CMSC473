# H1 Video Verification Quality Gate — Commit 5

## Purpose

This commit introduces an offline, fail-closed consistency audit for
BasketEvent video download reports and preliminary visual-verification
records.

The quality gate is designed to prevent unsupported verification
claims from being interpreted as evidence that a clip is ready for
computer-vision feature extraction or shot-prediction modeling.

It complements, but does not replace, independent manual visual
verification.

## Inputs

The audit uses two CSV files:

1. video_download_report.csv
2. visual_verification_v2.csv

Records are joined by exact NBA shot ID.

The script validates:
- Required CSV columns and unique shot IDs.
- Agreement on shot ID, game ID, event ID, and video path.
- Consistency of download statuses and file sizes.
- Explicit boolean verification fields.
- Nonnegative release-frame indices and valid intervals.
- Contradictory or unsupported verification claims.
- Missing download or review records.
- Claims of training eligibility without required evidence.

The original input files are not modified.

## Eight-clip pilot results

The updated audit reported:

- Download records: 8
- Review records: 8
- Shots audited: 8
- Preliminary unverified: 6
- Provisional release interval unverified: 2
- Invalid records: 0
- Training-eligibility claims: 0
- Claims not independently approved: 0
- Independently approved training clips: 0

The previous visual-review records identify candidate release intervals
for shots nba:0022500003:317 and nba:0022500032:283.

Neither release interval is conclusively verified.

A consistency-audit pass does not establish video-event identity or
precise release timing.

## Fail-closed eligibility policy

The script distinguishes three different concepts:

1. A reviewer's training-eligibility claim.
2. A claim requiring independent verification.
3. Independently approved training eligibility.

The current quality gate cannot independently verify visual content,
so it never certifies positive training eligibility.

Even a record containing plausible evidence text, release coordinates,
and player boxes is not automatically approved.

A verified status in an annotation manifest is not sufficient proof
that the corresponding video contains the intended NBA shot event.

## Automated tests

Twenty-five tests passed for the Commit 5 audit.

These cover:
- Preliminary and provisional records.
- Cross-report metadata inconsistencies.
- Invalid and incomplete release-frame intervals.
- Malformed boolean values.
- Duplicate and missing shot records.
- Invalid download statuses and file sizes.
- Contradictory verification claims.
- Positive eligibility claims that must remain unapproved.

The tests use synthetic fixtures and do not require videos, GPU models,
or network access.

## Limitations

This audit checks consistency of supplied metadata and review
decisions; it does not inspect actual MP4 frames.

It cannot independently confirm:
- That the correct NBA shot appears in the video.
- Exact ball-release timing.
- Shooter and defender identities.
- Physical defender distance.
- Automated tracking reliability.
- Whether CV features improve prediction performance.

A successful download record with positive file size is not equivalent
to independently verified frame decoding.

The current positive-eligibility evidence fields accept reviewer
statements and should not be interpreted as independent corroboration.

The current gate therefore reports zero independently approved clips.

The eight-clip pilot is not representative evidence of multi-game
CV feature coverage or predictive usefulness.

## Local reproduction

Run the audit using local reports:

    uv run --frozen python scripts/audit_h1_video_verification.py --downloads artifacts/h1_video_verification_8clips_v1/video_download_report.csv --reviews artifacts/h1_video_verification_8clips_v1/visual_verification_v2.csv --output artifacts/h1_video_verification_8clips_v1/verification_quality_gate_v3.json

The output path must not already exist.

Run the Commit 5 tests:

    uv run --frozen python -m pytest tests/test_audit_h1_video_verification.py -v

The CSV reports and MP4s are local experimental artifacts and are not
included in this commit. The real-data audit therefore cannot be
reproduced from a fresh clone without those local inputs.

## Next research milestone

Establish an independently reviewed video-event verification procedure
with explicit evidence for shot identity, release timing, and player
roles.

Only after sufficient independently verified clips are available
should the project evaluate automated feature extraction and
video-augmented prediction.

This commit makes no claim of CV-based prediction improvement.
