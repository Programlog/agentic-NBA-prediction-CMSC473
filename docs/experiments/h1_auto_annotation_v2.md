# H1 Experiment 11 — BasketEvent automatic release-frame estimator V2

## Motivation and preregistered comparison

V1 produced 36 proposals among 39 previously unannotated shots, but the *8-shot manually checked development sample* had **MAE 9 frames**, median absolute error **6**, exact matches **1/8**, and errors <=3 frames **4/8**. Largest errors: game 0022500003 shot 228 (29 frames), game 0022500004 shot 369 (17 frames), game 0022500032 shot 283 (12 frames). **These are V1 development results, not evidence for V2.**

V2 is an exploratory rule-based change. Rather than relying mainly on the maximum observed ball separation, it ranks separation *onsets* using a 3-frame near-ball baseline, local separation, sustained separation and upward movement. If no candidate survives, it explicitly falls back to V1 and labels the fallback. Candidates require manual verification. No outcome is used in release-frame scoring. The shot event label is used only to locate the annotated shooter in the JSON.

## Install

Copy the archive's `scripts/` and `tests/` and `docs/` entries to the project root. Keep `scripts/auto_annotate_basketevent.py` (V1) and all previous artifacts unchanged.

```powershell
uv run --no-sync pytest -q tests/test_auto_annotate_basketevent_v2.py
```

Run V2 on the **original manually annotated smoke manifest**, not on the V1-generated proposals, or it will preserve the proposals as though manually verified:

```powershell
uv run --no-sync python scripts/auto_annotate_basketevent_v2.py `
  --manifest artifacts/h1_multigame_video_v1/annotated_smoke.csv `
  --output artifacts/h1_auto_annotation_v2 `
  --development-reviews artifacts/h1_review_sample_v1.csv
```

Outputs: `auto_annotated_manifest.csv`, `annotation_audit.csv`, `annotation_summary.json`, and optional `development_comparison.json` for the original eight shots. Development comparison uses `abs(review_frame_delta)` as the V1 error and V2 candidate versus manually marked release as the V2 error. If V2 is worse, retain V1; do not claim improvement.

## Independent evaluation

The eight reviewed shots were available during V2 development. Any comparison on them is **in-sample diagnostic only**. For independent measurement, review additional shots that were not among those eight. The existing reviewer supports V2's exact artifact names:

```powershell
uv run --no-sync python scripts/review_basketevent_auto.py `
  --auto-dir artifacts/h1_auto_annotation_v2 `
  --input-reviewed artifacts/h1_review_sample_v1.csv `
  --output artifacts/h1_review_v2_holdout.csv `
  --sample-per-game 2
```

It skips the eight previously reviewed shots in the input-reviewed file. After review, run the comparison using the V2 audit and new reviewed file (or use the command below), and also compare V1 on *the same new shots* before claiming V2 generalizes. Ensure both algorithms' frame predictions are paired by `shot_id`; do not substitute `review_frame_delta` from V2 as V1's error. Preserve every skipped/rejected/unmatched item and per-game counts. Four games / a small shot sample are insufficient for strong model-accuracy claims.

## Safety & limitations

- Ball bounding boxes approximate ball location; missing or incorrect tracks degrade timing.
- The heuristic uses image geometry and does **not** infer physical release, confidence calibration, or pose.
- Shooter boxes and nearest-opponent proposals are not verified ground truth; nearest different jersey color can be wrong.
- Shooting side remains missing unless manually confirmed.
- Release-frame identification may inspect the full clip for annotation only; the shot prediction features **must** use only frames at or before the manually verified release. No outcome or post-release video features can enter XGBoost.
- Never overwrite V1 data, reviewed CSVs, or use auto proposals as model-ready annotations.
