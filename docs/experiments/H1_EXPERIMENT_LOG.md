# H1 Computer-Vision Experiment Log

**Updated:** November 20, 2026  
**Hypothesis H1:** Do video-derived shot mechanics and defensive context improve held-out shot-success predictions beyond traditional shot-log features?

This chronological record captures **successful, unsuccessful, and inconclusive** exploratory experiments. A completed pipeline run is not evidence that H1 is supported. Counts below come from contemporaneous local terminal output discussed with the experimenter, not a newly audited artifact bundle. Source manifests, raw clips, models, and some local result files have not yet been committed; preserve these for reproducibility.

## Experiment timeline

| Stage | Goal and setup | Observed outcome | Interpretation and next action |
| --- | --- | --- | --- |
| NBA video URL retrieval | Obtain clips for NBA shots via an initial video-fetch workflow | Initial 10 checked events returned **10/10 “No video URL returned”**; a separate 100-shot manifest existed | **Failed retrieval route**. Switched to BasketEvent, rather than treating empty results as a working data source |
| BasketEvent matching/pilot | Download and inspect BasketEvent event clips and pair them with shot events using game/event identifiers | Clips were accessible for game `0022500003`; an initial manually identified shot used `release_frame=125` and a shooter box; the pilot established a workable path to annotation | **Feasibility**, not an automatic release-frame or shooter-label guarantee. Keep manual annotations and provenance |
| YOLO11m pose extraction | Estimate shooter/defender pose-derived proxies on small manually annotated clips | Partial pilot results were generated; some clips yielded only shooter tracks, and defender information was often missing | **Mixed results**; do not treat every extracted feature as validated mechanics |
| YOLO11x pose extraction | Test larger pose weights on a ten-candidate batch | Local output counts: **5 ok, 4 partial, 1 error** | **Not a complete 10/10 successful pose batch**. Save exact errors, checkpoint provenance, and per-clip comparison before claiming model improvement |
| CoTracker3 extraction (10-shot) | Track a 3×3 release-frame point grid per shooter/annotated defender across 12 preceding frames and release | **10/10 extraction runs succeeded**. Mean shooter visibility **80.09%** across 10; defender visibility **49.72%** across 6 annotated defenders. At least five points visible at both endpoints: shooter **8/10**, defender **2/6** | **Feasible tracking but weak defender visibility**. Image-point displacement is not verified player movement |
| Static trajectory audit | Overlay point paths on one release-frame PNG per clip | **10/10 PNGs generated**, but crowded trajectories and overlapping players were difficult to interpret; this separate visualizer originally used 20/50/80% seeds instead of extractor's 25/50/75% | **Audit limitation discovered**; fix point seed mismatch and render frame-by-frame video |
| CoTracker3 frame-by-frame visual audit V2 | Use shared 25/50/75% seed grid and draw per-frame MP4 trails | **10/10 MP4s saved**. Visual review flagged player overlap, possible point drift and broadcast-camera motion | **Qualitative audit only**, not verified tracking ground truth. Review especially shots 58, 535 and 579 |
| Camera-motion correction: two-shot pilot | Background Lucas–Kanade flow + translation-only correction on shots 579 and 600 | **2/2 processed** and **2/2 internal camera estimates valid**; shooter normalized displacements increased from 0.017553 to 0.018002 (579) and from 0.015158 to 0.016172 (600) | **Feasible correction, unknown accuracy**; correction need not reduce displacement |
| Camera-motion correction: ten-shot follow-up | Same 12-frame lookback, max side 384 px, CPU; all shots from game `0022500003` | **10/10 processed**, **10/10 internal camera estimates valid**. Mean shooter displacement **0.02473 → 0.02814**, mean defender **0.04035 → 0.04395** for six defenders | **Inconclusive accuracy benefit**. Five of six annotated defenders flagged low visibility. See [full camera-motion results](h1_camera_motion_results.md) |

## Detailed result: CoTracker3 ten-shot extraction

- Input manifest: `data/basketevent/batch_01_manifest.csv`.
- All ten shots from **one game**, so these results are **not** independent multi-game testing.
- CoTracker3 offline inference, CPU, release-anchored 3×3 interior seeds, 12-frame lookback, 384-pixel max image side.
- Earlier tracker extraction metrics: shooter mean visibility **80.09%**, defender mean visibility **49.72%** (only six defenders annotated).
- Endpoints: **8/10 shooters** versus **2/6 defenders** had at least 5/9 points visible at both ends.
- An initial static visualizer used a **different** seed grid (20/50/80% rather than 25/50/75%); its visibility outputs must **not** be treated as interchangeable with those from the extraction script. V2 fixed this mismatch for subsequent video audits.
- **Limitation:** high point visibility does not establish consistent player identity; image-point motions may reflect camera movement.

## Detailed result: camera-motion follow-up

The follow-up combines CoTracker3 with optical flow of nominally background corners to estimate a per-frame median **screen translation**.

- 10/10 successful; 10/10 internal validity checks passed.
- Mean camera motion normalized to the image diagonal: **0.00772**.
- Mean shooter displacement: **0.02473 raw → 0.02814 corrected** across ten shots.
- Mean defender displacement: **0.04035 raw → 0.04395 corrected** across six annotated defenders.
- Tracking-quality flags: shooter **7 review_required, 2 low_visibility, 1 few_endpoint_points**; defender **1 review_required, 5 low_visibility, 4 no_annotation**.
- Correction did **not consistently reduce displacement** and has **not been validated against stationary landmarks or player ground truth**.
- The data supports **technical feasibility**, not improved feature accuracy or improved shot prediction. Full table, commands, limitations and pilot: [camera-motion report](h1_camera_motion_results.md).

## What failed or remains unresolved?

1. **Shot URL retrieval failed** on the initial tested items, requiring a different clip source.
2. **YOLO pose extraction was incomplete** on some shots, including a larger-model ten-shot run with one error and four partial outputs.
3. **Defender tracking has high missingness and poor visibility**. The camera-correction follow-up marks five of six annotated defenders low-visibility.
4. **Static trajectory plots were not adequate** to visually debug temporal tracking; frame-by-frame MP4 audits were added.
5. **Camera correction quality is unverified**. The internal flow consistency test cannot distinguish stationary landmarks from coherently moving non-stationary image content.
6. **H1 predictive evaluation is outstanding**: no matched multi-game, held-out XGBoost comparison with the final CV/corrected features has been reported here.

## Reproducibility checklist for every future H1 experiment

Before interpreting a result, record:

1. Experiment identifier and run date/time.
2. Question and hypothesis, expected failure modes, and acceptance criteria fixed before inspecting outputs.
3. Git commit/branch, complete command, Python/torch/OpenCV versions, model names/weights and hashes.
4. Source dataset, video provenance, shot/event IDs, game IDs, annotation manifest version/hash, and split policy.
5. Number attempted, succeeded, partial, failed, missing annotations, and reason per failure.
6. Output file paths, CSV/JSON reports, example audit images/videos, numerical summaries and uncertainty when applicable.
7. Interpretation: separate **execution**, **feature validity**, and **predictive performance**; discuss negative results.
8. Specific follow-up experiment informed by the finding.

## Next milestones

- Preserve local H1 artifact CSVs/manifests and collect precise error logs from the early failed/partial extraction runs.
- Verify camera translation against manually selected stationary court landmarks; test zoom/perspective robustness.
- Improve occlusion handling, identity retention, and missing-defender behavior.
- Expand to multiple games with outcome-blind annotation/feature extraction.
- Freeze a held-out game split and evaluate tabular XGBoost versus tabular + H1 features on exactly matched shots, with Brier score, log loss, calibration and confidence intervals. Avoid silently restricting evaluation to successes only.

**Current scientific status:** H1 feature extraction and visual audits are feasible on a small pilot, but **H1 has not yet been tested decisively**.
