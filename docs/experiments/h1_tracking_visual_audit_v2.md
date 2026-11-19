# H1 CoTracker3 frame-by-frame audit (v2)

This adds **per-shot MP4 videos** that display visible point tracks over the
13-frame pre-release window. Each video shows green shooter points, red defender
points, and short point histories. The annotated release-frame bounding boxes
appear **only on the last frame**: their locations were not independently
verified at earlier times.

Crucially, this visualizer **imports the same `seed_queries` function** as
`h1_motion_tracker.py`, using 25%, 50%, 75% release-frame box positions.
It replaces the earlier standalone visualizer's 20%, 50%, 80% grid.

## Install

Copy `src/nba_shots/h1_track_visualizer.py`,
`tests/test_h1_track_visualizer.py`, and this guide into the matching project
folders. The script *replaces* the previous `h1_track_visualizer.py`. It does
not modify the existing `h1_motion_tracker.py` or its saved experiment outputs.

## Tests

```powershell
uv run --no-sync pytest -q tests/test_h1_track_visualizer.py tests/test_h1_motion_tracker.py
```

## Run on the same 10-shot manifest

```powershell
uv run --no-sync python -m nba_shots.h1_track_visualizer `
  --manifest data/basketevent/batch_01_manifest.csv `
  --output artifacts/cotracker3_batch_10_audit_v2 `
  --limit 10 `
  --lookback 12 `
  --max-side 384 `
  --device cpu `
  --trail 5
```

Outputs: `videos/*.mp4`, `tracking_audit.csv`, and `summary.json`.
To play the first shot:

```powershell
start artifacts\cotracker3_batch_10_audit_v2\videos\nba_0022500003_579.mp4
```

**Limitations:** This re-runs CoTracker3, rather than reusing saved tracks,
so values can differ from older runs; inspect its `tracking_audit.csv`.
A green/red dot means the tracker estimates a visible image point, **not**
that correct player identity was verified. Camera panning, occlusion, and drift
remain unresolved; do not treat video-space motion as physical speed or claim
prediction improvements. All input frames are at or before annotated release.
