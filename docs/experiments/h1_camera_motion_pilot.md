# H1 camera-motion correction pilot (experimental)

This add-on is separate from the established H1 `h1_motion_tracker.py` and CoTracker3 audit videos. It does not overwrite the original 10-shot results or change the existing CV feature schema.

## Method

1. Read only frames from `release_frame - lookback` through the annotated release frame (inclusive).
2. Seed 9 shooter points and, when annotated, 9 defender points using the *exact* `h1_motion_tracker.seed_queries` 25%/50%/75% grid.
3. Run CoTracker3 offline with backward tracking, as in the original extractor.
4. On the release frame, find high-contrast background corners with OpenCV, excluding padded shooter/defender boxes and a frame boundary band. Track those corners backward using Lucas–Kanade optical flow.
5. Estimate a **global translational camera shift** via the median background-point displacement on each adjacent frame pair. Accumulate shifts forward through the 13-frame window. Subtract the inferred shift from CoTracker3 point locations.
6. Compute raw versus corrected shooter/defender displacement and step-length proxies (both frame-diagonal normalized). Record visibility and heuristic tracking flags.

## Run (Windows PowerShell)

After unpacking from the repo root, test:

```powershell
uv run --no-sync pytest -q tests/test_h1_camera_motion.py
```

For the first 2 shots:

```powershell
uv run --no-sync python -m nba_shots.h1_camera_motion `
  --manifest data/basketevent/batch_01_manifest.csv `
  --output artifacts/h1_camera_motion_pilot_2 `
  --limit 2 `
  --lookback 12 `
  --max-side 384 `
  --device cpu
```

Then run all 10 in a **new output folder**:

```powershell
uv run --no-sync python -m nba_shots.h1_camera_motion `
  --manifest data/basketevent/batch_01_manifest.csv `
  --output artifacts/h1_camera_motion_batch_10 `
  --limit 10 `
  --lookback 12 `
  --max-side 384 `
  --device cpu
```

Review:

```powershell
Get-Content artifacts/h1_camera_motion_batch_10/summary.json
Import-Csv artifacts/h1_camera_motion_batch_10/camera_motion_audit.csv | Format-Table shot_id,status,camera_valid,shooter_tracking_flag,defender_tracking_flag -AutoSize
```

## Interpretation

- `camera_valid`: only true when enough background point tracks survive **every** step, with low enough dispersion to support a simple shift estimate.
- `camera_seed_points`, `camera_min_flow_points`, `camera_max_flow_mad_px`: diagnostics for the background tracks.
- `camera_total_displacement_norm`: estimated camera translation magnitude divided by frame diagonal.
- `cotracker_*`: **raw** point movement using the same mathematical definition as the original extractor.
- `corrected_cotracker_*`: movement after subtracting estimated translation. Missing/invalid corrections stay **NaN**, never zero.
- `*_tracking_flag`: preliminary manual-review prioritization. `review_required` is **not** an assertion that tracking is accurate. Fewer than five endpoint-visible points or <60% visibility lead to warning flags.

## Limitations and safeguards

- Broadcast camera pans, zooms, rotation, perspective changes, and motion parallax are **not** fully explained by one 2D translation.
- The automatic background mask removes annotated people only. Other moving players and moving overlays can pollute the background corner set. Review actual videos and diagnostics.
- Backward optical flow is an estimate and is **not** validated against ground-truth camera motion.
- This analysis may rerun CoTracker3 inference. No pre-existing `artifacts/` output is overwritten.
- All 10 clips in this pilot are from one game. These features are **not** validated shot-prediction improvements; do not train a performance claim on this 10-shot sample.
- We have only unit-tested the script on synthetic examples. Real BasketEvent processing must be verified on the user's machine.
