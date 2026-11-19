# H1 CoTracker3 pre-release motion pilot

## Research purpose

H1 asks whether computer-vision-derived shooting/defender features improve
NBA shot-success predictions beyond traditional shot logs. Existing
`nba-shots extract-cv` uses YOLO-Pose at the annotated release frame.
This **experimental, separate** CoTracker3 pilot measures movement of
annotated image points **before** release. It does not yet train XGBoost,
produce an H1 verdict, or replace the team's existing CV feature schema.

[CoTracker3](https://github.com/facebookresearch/co-tracker) is a pretrained
point tracker, **not** a basketball-specific player detector, skeleton estimator,
or ball tracker. We seed points **inside the annotated release-frame shooter and
optional defender bounding boxes** and track them backward over a short
pre-release window. Motion can include camera movement, occlusion, or person
mix-ups; manually audit results before treating any proxy as meaningful.

## Inputs and leakage safeguards

Use an already annotated manifest containing:

- `shot_id` (unique)
- `clip_path` (absolute, or relative to the manifest's parent folder)
- `release_frame` (zero-based)
- `shooter_box` (JSON `[x1,y1,x2,y2]`)
- `defender_box` (same JSON format, or empty)

The pilot decodes **only** frames `[max(0, release_frame - lookback), release_frame]`.
The release frame is the **last frame supplied to CoTracker3**. Inference uses
`queries=[t_release,x,y]` with `backward_tracking=True`, so it does not see
follow-through, ball entry, makes/misses, or other post-release information.

This tool still assumes the original clip and the release-frame annotation
identify the correct shooter; it does not independently validate a play match.

## Environment

This is an optional heavyweight CV experiment. Install PyTorch and TorchVision
using the platform-specific instructions at
<https://pytorch.org/get-started/locally/>, along with the existing repo's
vision dependencies. CoTracker3 is loaded via official PyTorch Hub (first run
may download source and checkpoint). Pin the upstream version/checkpoint for
strict reproducibility before a larger study.

For example, after `uv sync --frozen --all-extras` and installing compatible
PyTorch/TorchVision into the same `.venv`, run from the repository root:

```powershell
uv run --no-sync python -m nba_shots.h1_motion_tracker `
  --manifest data/basketevent/batch_01_manifest.csv `
  --output artifacts/cotracker3_pilot_2 `
  --limit 2 `
  --lookback 12 `
  --max-side 384 `
  --device cpu
```

On a CUDA-capable machine you may choose `--device cuda`. The user's Intel UHD
Graphics 620 system has previously reported CPU-only Ollama inference;
**CoTracker3 can run on CPU but may be slow**, so test one or two annotated
shots first or use a separate compatible GPU machine. CPU inference time and
model quality have not yet been measured on this project's clips.

## Outputs

`artifacts/cotracker3_pilot_2/cotracker_features.csv` records one row per attempted
shot, including failures and missing defender annotations. For each annotated
person it records:

- `cotracker_{shooter,defender}_points`: 9 release-frame seeds per person
- `cotracker_{shooter,defender}_visibility_fraction`: fraction of tracked
  point-frames the model marks visible
- `cotracker_{shooter,defender}_endpoint_visible_points`: number visible at
  both first and final input frames
- `cotracker_{shooter,defender}_median_displacement_norm`: median displacement
  from the earliest to release frame, divided by frame diagonal
- `cotracker_{shooter,defender}_median_step_norm`: median visible adjacent-frame
  step length, divided by frame diagonal

`summary.json` reports attempted, succeeded, and failed shots, settings, and a
warning that these are **2D image-space motion proxies**, not physical release
angles, velocities, defender contest heights, or ball trajectories. Missing
measurements are left missing; failed shots are not silently dropped.

## Next evaluation steps

1. Inspect clips and point tracks for shooter/defender correctness and camera pan.
2. Add explicit background-point motion compensation if motion is dominated by
   broadcast camera movement; do not interpret screen motion as player motion.
3. Repeat on a fixed annotated cohort without selecting shots by outcomes or
   tracking success.
4. Only then connect reliable features to the existing H1 matched-cohort
   XGBoost baseline and CV-enhanced comparison, freezing split and protocol.
5. Measure feature coverage and compare Brier score, log loss, ROC-AUC,
   and accuracy **on exactly the same held-out games**.

**Status:** implementation and isolated unit tests added. No real-video
CoTracker3 model inference results or shot-prediction improvements are claimed.
