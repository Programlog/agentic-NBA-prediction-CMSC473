# H1 Experiment 9 — Automatic BasketEvent annotation proposals (V1)

## Objective

Use the 40-shot, 4-game multi-game cohort to generate *unverified* release-frame,
shooter, and defender proposals from BasketEvent's existing player and ball
bounding-box trajectories. Preserve the existing human-verified annotation.

This is a proposal-generation experiment, **not** evidence of accurate automatic
ball-release detection or improved shot prediction.

## Inputs and provenance

- `artifacts/h1_multigame_video_v1/annotated_smoke.csv` (contains the first
  manually verified shot, if saved successfully).
- `artifacts/h1_multigame_video_v1/event_matching_report.csv`: exact shot-to-
  shooter-role matching evidence, produced by the multi-game retrieval script.
- `data/basketevent/`: original clip MP4s and JSONs, same relative paths as
  `source_json` in the manifest. JSON trajectories use `[x,y,width,height]`.

Do not change or overwrite these source files. Game IDs are read as strings.

## Command (PowerShell, repository root)

```powershell
uv run --no-sync pytest -q tests/test_auto_annotate_basketevent.py

uv run --no-sync python scripts/auto_annotate_basketevent.py `
  --manifest artifacts/h1_multigame_video_v1/annotated_smoke.csv `
  --output artifacts/h1_auto_annotation_v1
```

## Algorithm

1. Use `event_matching_report.csv` to select a unique audited shooter role;
   verify it corresponds to a made/missed shot event. Neither outcome nor event
   clock is used to infer release timing.
2. Verify each MP4's frame count matches shooter and ball trajectories.
3. Convert player/ball `[x,y,w,h]` records to bounded `xyxy` pixel boxes.
4. Score candidate separation events when the ball was near the shooter's
   approximate upper torso before the frame, moves upward, and moves away
   afterward. Output up to five candidate frame indices and heuristic scores;
   these are *not* calibrated confidence values.
5. At the top candidate frame, propose the shooter box. Propose the nearest
   different-jersey-color person as the defender only when within a cautious
   distance threshold. This is not reliable opponent identification.
6. Leave `shooting_side` blank: a trajectory does not identify the shooting
   hand. Preserve every row with a preexisting release-frame annotation.
7. Record all errors and missing candidates, rather than deleting rows.

## Outputs

- `auto_annotated_manifest.csv` — all 40 shots, existing verified annotations
  retained verbatim, proposed values marked `auto_proposed_review_required`.
- `annotation_audit.csv` — candidate frame lists, uncalibrated heuristic scores,
  source shooter role, defender role and rejection reasons, failures.
- `annotation_summary.json` — counts and limitations.

## Scientific limitations and next checks

- Ball tracking may fail through occlusion, and camera motion or passing can
  imitate release. This heuristic is *not* validated against a labelled
  release-frame dataset. Its top candidate can be wrong.
- Scoring uses post-candidate frames **only to propose the release label**, not
  as model features. The eventual CV extractor must use only frames at or before
  the **manually verified** release frame.
- Player box identity and player/ball frame alignment also need visual audits.
- The `shooting_side` field intentionally stays missing for automated rows.
  Existing pose extraction is not ready until a human confirms it.
- Report candidate recall / frame error using a blinded manual sample from
  multiple games *before* deciding whether to expand use of auto proposals.
- **Do not train XGBoost, YOLO, or CoTracker3 from auto-proposed rows as though
  they were verified.** Review the flagged cases first.

## Results

Pending execution on the user's local downloaded MP4s. No extraction accuracy
or H1 predictive benefit is claimed.
