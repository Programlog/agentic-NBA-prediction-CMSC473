# H1: Preliminary Video and Manual Annotation Feasibility Audit

## Scope

This report records a small, exploratory H1 computer-vision feasibility
assessment. It does not establish that the video candidates are suitable
for training or that visual features improve NBA shot prediction.

Eight BasketEvent MP4 clips were selected from four NBA games for
manual visual review. All eight were downloaded successfully and
included in the preliminary verification log.

The original 40-shot matching manifest was not modified.

## Video verification results

- Clips reviewed: 8
- Represented NBA games: 4
- Successful video downloads: 8
- Conclusively verified video-event identities: 0
- Confirmed exact release frames: 0
- Clips marked CV-training eligible: 0
- Clips with provisional release intervals: 2

Shot nba:0022500003:317:
- Candidate release interval: frames 120-122.
- The Cleveland #31 layup appears visually consistent with its
  metadata, but a broadcast-timing discrepancy remains unresolved.
- Event identity and exact release are not conclusively verified.

Shot nba:0022500032:283:
- Candidate release interval: frames 100-102.
- The San Antonio #1 three-point shooting motion is visible.
- Motion blur prevents confirming an exact release frame.
- The broadcast clock is near 3:54, versus metadata clock 3:53.
- Event identity and exact release remain unconfirmed.

The other six clips have no confirmed release interval in the
verification log. Several present congestion, visibility, or
event-identification challenges.

## Manual pre-release annotation pilot

Shot: nba:0022500032:283

Image dimensions: 1280 x 720 pixels.

Three earlier frames were annotated: 85, 90, and 95.

Each frame contains three manual player-role boxes:
- Shooter candidate: Wembanyama #1
- Candidate defender: Houston #12
- Candidate defender: Houston headband player

The revised frame-90 annotation was used in the final local audit.

The updated validation audit reported:
- Expected bounding boxes: 9
- Valid bounding boxes: 9
- Image-space center-distance measurements: 6
- Validation errors: 0
- Validation passed: true

Thirteen automated unit tests passed, including tests for invalid
coordinates, duplicate annotations, missing roles, incorrect shot IDs,
frame-to-file mismatches, and unexpected frame records.

These tests verify data structure and computation, not visual accuracy
or player-identity correctness.

## Measurement interpretation

The computed distances use the centers of manually drawn bounding
boxes and are expressed in image pixels.

They are not physical distances in feet. Changes in camera perspective,
player pose, arm extension, and annotation boundaries can affect them.

The primary contesting defender has not been conclusively identified.
The proposed release interval has not been independently confirmed.

These annotations must not be treated as validated, outcome-independent
features for a shot-prediction model.

## Coverage and limitations

Only one shot has manually annotated player boxes across multiple
pre-release frames. The eight-video review does not establish
multi-game feature extraction coverage.

No automatic detector or tracker was evaluated here. No comparison
between tabular-only and video-augmented predictive models was
performed.

The manual annotation CSV files and video evidence are local experiment
artifacts. A separate tracked sample or fixture would be necessary to
make the real-data audit independently reproducible from a fresh clone.

## Local reproduction

With the local annotation CSVs available:

    uv run --frozen python scripts/audit_h1_manual_boxes.py --annotations artifacts/h1_video_verification_8clips_v1/manual_annotations --output artifacts/h1_video_verification_8clips_v1/manual_box_quality_audit_v3 --shot-id nba:0022500032:283 --width 1280 --height 720 --frames 85 90 95

Unit tests:

    uv run --frozen python -m pytest tests/test_audit_h1_manual_boxes.py -v

The output path must not already exist because the audit script
intentionally refuses to overwrite an earlier result.

## Next research milestone

Before using video-derived features for predictive modeling, the
project needs independently checked event identity and release timing,
a consistent definition of player visibility and defender assignment,
and broader cross-game coverage measurements. Automatic detection
and tracking should be evaluated against manually reviewed references.

No CV-based prediction improvement is claimed by this audit.
