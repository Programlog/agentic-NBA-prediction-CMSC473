# H1 Camera-Motion Compensation: Two-Shot Pilot and Ten-Shot Follow-up

**Status:** completed exploratory extraction and diagnostics; **not** an H1 shot-prediction improvement result.

**Reporting date:** November 20, 2026 (compiled from local terminal output shared by the experimenter).

## Research question

Can background optical flow estimate broadcast-camera translation within a short pre-release window, and how does subtracting that estimate change CoTracker3 player motion features?

This is a feature-quality investigation, not a test of whether the resulting features improve XGBoost Brier score. The project H1 hypothesis remains open.

## Context and motivation

CoTracker3 was applied to 10 release-annotated BasketEvent shots from one game (`0022500003`). The earlier 10-shot CoTracker3 pilot reported 10/10 successful extractions, but its point visibility was substantially weaker for defenders than for shooters. Dense scene overlap and broadcast-camera movement can affect 2D point displacement.

This follow-up tests a **translation-only** correction based on background Lucas–Kanade optical flow. It preserves uncorrected measurements and adds corrected values without replacing the original H1 feature schema.

## Experiment configuration

| Setting | Value |
| --- | --- |
| Source | Local BasketEvent clips and annotated `data/basketevent/batch_01_manifest.csv` |
| Game | `0022500003` (all 10 shots from the same game) |
| Model | CoTracker3 offline via PyTorch Hub |
| Motion inputs | 9 release-frame seed points per annotated shooter; 9 per annotated defender |
| Temporal window | 12 frames before release + annotated release frame (13 frames) |
| Maximum image side | 384 px |
| Device | CPU |
| Camera estimate | Background good-features-to-track corners, Lucas–Kanade flow, per-frame median translation |
| Background point budget | 180 |
| Excluded frames | All post-release frames |
| Script | `src/nba_shots/h1_camera_motion.py` (local experimental script; GitHub availability not asserted) |
| Local outputs | `artifacts/h1_camera_motion_pilot_2/` and `artifacts/h1_camera_motion_batch_10/` |

**Run command (PowerShell)**

```powershell
uv run --no-sync python -m nba_shots.h1_camera_motion `
  --manifest data/basketevent/batch_01_manifest.csv `
  --output artifacts/h1_camera_motion_batch_10 `
  --limit 10 `
  --lookback 12 `
  --max-side 384 `
  --device cpu
```

Local output files: `summary.json` and `camera_motion_audit.csv`.

## Pilot: two shots

The pilot evaluated shots `nba:0022500003:579` and `nba:0022500003:600`. **2/2 processed; 0 errors; 2/2 internal camera-validity checks passed.**

| Shot | Background seeds | Minimum flow points | Maximum flow MAD (px) | Camera displacement (normalized) | Shooter raw | Shooter corrected | Defender raw | Defender corrected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 579 | 180 | 180 | 0.044687 | 0.001356 | 0.017553 | 0.018002 | 0.052833 | 0.051817 |
| 600 | 180 | 180 | 0.248785 | 0.005543 | 0.015158 | 0.016172 | N/A | N/A |

Both shooter displacements increased slightly after correction. The annotated defender's displacement on shot 579 decreased slightly. **A positive or negative change is not itself evidence of improved quality.**

## Ten-shot follow-up: extraction result

The same pipeline processed **10/10 shots successfully**, with **0 extraction failures**, and marked camera correction valid for all 10 using its **internal** consistency checks. That validates pipeline execution, not ground-truth correction quality.

## Ten-shot follow-up: observed motion features

All values are normalized by the image diagonal, not feet, meters, or real-world speed.

| Shot suffix | Camera motion | Shooter raw | Shooter corrected | Defender raw | Defender corrected |
| --- | ---: | ---: | ---: | ---: | ---: |
| 579 | 0.00136 | 0.01755 | 0.01800 | 0.05283 | 0.05182 |
| 600 | 0.00554 | 0.01516 | 0.01617 | N/A | N/A |
| 168 | 0.00347 | 0.01063 | 0.00835 | 0.00755 | 0.00411 |
| 371 | 0.00282 | 0.00561 | 0.00651 | 0.02238 | 0.01980 |
| 481 | 0.00842 | 0.01845 | 0.02669 | N/A | N/A |
| 535 | 0.00645 | 0.03395 | 0.03830 | 0.09137 | 0.09778 |
| 581 | 0.00547 | 0.03852 | 0.03784 | N/A | N/A |
| 105 | 0.00686 | 0.05248 | 0.05607 | 0.03686 | 0.04052 |
| 58 | 0.02047 | 0.04339 | 0.06099 | 0.03113 | 0.04968 |
| 382 | 0.01636 | 0.01158 | 0.01252 | N/A | N/A |

**Means from the local CSV, as printed by the experimenter:**

| Quantity | Mean | Coverage |
| --- | ---: | --- |
| Estimated camera displacement | 0.00772 | 10 shots |
| Shooter displacement, raw | 0.02473 | 10 shots |
| Shooter displacement, corrected | 0.02814 | 10 shots |
| Defender displacement, raw | 0.04035 | 6 annotated defenders |
| Defender displacement, corrected | 0.04395 | 6 annotated defenders |

Based on these rounded means, average shooter displacement increased approximately **13.8%** after correction, and average defender displacement increased approximately **8.9%**. **8 of 10** corrected shooter values increased; **3 of 6** defender values increased. The largest estimated camera motion was shot `58` (0.02047), where both corrected displacements were larger.

**Tracking-quality flags (exploratory thresholds):**

| Flag | Shooter (10 shots) | Defender (10 shots) |
| --- | ---: | ---: |
| `review_required` | 7 | 1 |
| `low_visibility` | 2 | 5 |
| `few_endpoint_points` | 1 | 0 |
| `no_annotation` | 0 | 4 |

`review_required` **does not mean verified-good tracking**. For defenders, only six shots had annotations; five of those six were flagged low visibility.

## Interpretation

- The local implementation performed camera-translation estimation without extraction errors on the entire 10-shot pilot; all estimates passed its own validity criteria.
- Translation removal changes displacement in either direction; it is **not** a procedure to always make motion smaller.
- The correction **did not consistently reduce image-space displacement**. Neither an increase nor decrease establishes physical accuracy.
- Defender tracking remains an important bottleneck.
- No new XGBoost training/evaluation was performed; **H1 is not yet supported or rejected by this experiment**.

## Failure modes and limitations

1. **Validity is not ground truth:** internally consistent optical flow may track moving background players or scene elements that are not stationary.
2. **Translation-only:** zoom, rotation, perspective, and parallax are not corrected.
3. **Small, one-game sample:** 10 shots from the same game cannot establish generalizable results or valid between-game predictive gains.
4. **Defender missingness:** four shots lack defender annotations; five of the six others were flagged for low tracking visibility.
5. **No physical units:** normalized 2D image displacement cannot be interpreted as real-world speed.
6. **No paired predictive comparison:** no frozen matched-game XGBoost baseline versus corrected-feature test yet.
7. **Local artifacts:** `camera_motion_audit.csv` and raw clips remain in local `artifacts/` and `data/`; this report transcribes displayed terminal outputs rather than bundling raw artifacts or hashes. Record code commit, checkpoint digest, and source hashes before a larger replication.

## Reproducibility and next actions

1. Preserve the local CSV, summary, exact annotated manifest, clip IDs, code revision, and checkpoint provenance.
2. Inspect per-frame MP4 tracking audits and verify that background points are actually stationary court landmarks.
3. Compare translation-only against a robust affine/homography estimate and quantify residual background motion, ideally on independent manual landmark annotations.
4. Improve defender tracking, especially occluded/overlapping players.
5. Expand to multiple games, freeze splits by game, and evaluate original versus CV-enhanced XGBoost on identical held-out shots. Do not select based on shot outcomes or only successfully tracked examples.

**Conclusion:** The camera-correction experiment was technically feasible on all 10 local shots, but it has **not demonstrated an improvement in motion-estimation accuracy or shot-success prediction**.
