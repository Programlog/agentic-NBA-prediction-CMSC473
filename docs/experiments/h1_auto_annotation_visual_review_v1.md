# H1 Experiment 10 — BasketEvent automatic annotation visual review v1

## Purpose
Validate proposed release frames/boxes from **Experiment 9** before CV modeling. The proposed release-frame estimator is heuristic; a produced candidate is **not** evidence of correct frame timing. Source clips contain outcome and post-release footage: those may assist labeling, but predictive features must be taken at/before a **manually verified** release frame.

## Inputs and provenance
- `artifacts/h1_auto_annotation_v1/auto_annotated_manifest.csv`
- `artifacts/h1_auto_annotation_v1/annotation_audit.csv`
- `artifacts/h1_auto_annotation_v1/annotation_summary.json` (records original manifest path)
- Original `artifacts/h1_multigame_video_v1/annotated_smoke.csv` and its nearby `event_matching_report.csv`
- `data/basketevent/test/<game>/*.json` and corresponding MP4s

**Paths:** MP4 relative paths in the auto manifest were originally resolved relative to the original manifest, not relative to the auto-directory. This reviewer explicitly resolves them relative to the original manifest.

## Start with a balanced review sample (two *unverified* shots per represented game)

```powershell
uv run --no-sync pytest -q tests/test_review_basketevent_auto.py
uv run --no-sync python scripts/review_basketevent_auto.py `
  --auto-dir artifacts/h1_auto_annotation_v1 `
  --output artifacts/h1_review_sample_v1.csv `
  --sample-per-game 2
```

The script opens each MP4 at the top suggested release frame, or its midpoint if there was no candidate. It overlays BasketEvent player boxes and ball box, highlights the shooter, and initially highlights the automatically proposed defender (which must be checked). Switch between suggested frames using `C`/`V`, move by 1 using `A`/`D`, or by 10 using `J`/`L`. **You must press `R` to record the correct release frame and `1`/`2` to set hand before `S` can save.** Click on a player to choose a defender or press `X` to leave defender missing. Press `N` to explicitly reject a bad/unusable clip, `K` to skip without changing its annotation, or `Q` to quit while preserving checkpoints.

Existing manually verified rows are skipped by default. The output CSV is written early and checkpointed after every saved/rejected shot; source files are never overwritten.

To continue from the sample with another **new output file**, use:

```powershell
uv run --no-sync python scripts/review_basketevent_auto.py `
  --auto-dir artifacts/h1_auto_annotation_v1 `
  --input-reviewed artifacts/h1_review_sample_v1.csv `
  --output artifacts/h1_review_remaining_v1.csv `
  --sample-per-game 0
```

Never rerun with an existing output filename; choose a new filename to preserve results.

## Quantify actual accuracy after reviewing

The reviewer writes `review_frame_delta = manually_verified_frame - top_auto_candidate` for each reviewed shot with a candidate; values near zero suggest accuracy, while positive/negative values indicate missed timing. For shots without a candidate, delta is empty. `review_candidate_frames`, `review_shooter_role`, and `review_defender_role` are saved for audit.

```powershell
uv run --no-sync python -c "import pandas as pd; d=pd.read_csv('artifacts/h1_review_sample_v1.csv'); r=d[d.annotation_status.eq('manual_review_verified_basketevent_boxes')]; e=r.review_frame_delta.dropna(); print('Reviewed:',len(r)); print('Candidate comparisons:',len(e)); print('Median absolute frame error:',e.abs().median() if len(e) else 'N/A'); print('Within 3 frames:',int(e.abs().le(3).sum()) if len(e) else 'N/A')"
```

**Interpreting quality:** This is a tiny preliminary audit, not an accuracy guarantee for all shots. Review cases from each game, including no-candidate cases. Box correctness, shooter identity, defender choice, release accuracy, and hand selection must be visually checked. This visual review relies on annotations provided in BasketEvent and is not independent ground truth for the player boxes.

## Modeling gate
Never use `auto_proposed_review_required`, `review_required_no_release_candidate`, or `manual_review_rejected` rows as **verified** CV features. Only reviewed frames/boxes may be used for a trusted extraction. Preserve missing/failed rows in the registered cohort and report coverage. Do not perform random shot-level train/test splitting across games or compare to a different-season Kaggle cohort.
