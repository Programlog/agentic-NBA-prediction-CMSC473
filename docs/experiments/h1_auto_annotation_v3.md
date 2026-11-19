# H1 Experiment 12 — V3 agreement-based release-frame review triage

## Motivation and prior measured results

Eight manually reviewed shots from four games showed V1 release-frame mean absolute error (MAE) **9.0 frames**, median **6.0**, with **4/8** within three frames. V2 development-set MAE was **20.5 frames**, median **2.0**, with **5/8** within three frames. V2's error for `nba:0022500003:317` was **138 frames** (V1: 1 frame); V1 proposed frame 120, V2 frame 259 and manual release was 121. Neither method is independently validated on unseen shots.

V3 is **not a third ball-release estimator**. It reconciles the frozen outputs of both estimators, avoids false assurance on disagreements, and prioritizes manual review. No new frames, trajectory evidence, or shot outcomes are used in the decision rule.

## Decision rule (predeclared threshold: 5 frames)

- Both predictions present and difference ≤ 5: `agreement_candidate_unverified`. The audit records V1's candidate as a *review hint*; **the output manifest intentionally leaves release frame and player boxes blank** until human verification.
- Both predictions present and difference > 5: `review_required_disagreement`; no release frame is promoted.
- One present: `review_required_single_estimator`.
- Neither present: `review_required_no_candidate`.
- Manually verified annotations from a prior review CSV take precedence. Next priority: manually verified rows from V1/V2 manifests. Manual rejections are also preserved; auto values can never override manual decisions.

The reconciliation output is **not a usable training manifest** for unreviewed shots. Unverified rows keep empty release frame, boxes and shooting side. An existing review checkpoint is a development reference and not an independent validation set.

## Installation

Copy ZIP contents into the repository root. Run from the project root:

```powershell
uv run --no-sync python -m pytest -q tests/test_reconcile_basketevent_v3.py
```

## Run V3 on the existing 40-shot cohort

```powershell
uv run --no-sync python scripts/reconcile_basketevent_v3.py `
  --v1-dir artifacts/h1_auto_annotation_v1 `
  --v2-dir artifacts/h1_auto_annotation_v2 `
  --reviews artifacts/h1_review_sample_v1.csv `
  --output artifacts/h1_auto_annotation_v3 `
  --threshold 5
```

Outputs (new directory only): `v3_review_manifest.csv`, `v3_reconciliation_audit.csv`, and `v3_summary.json`. `v3_reconciliation_audit.csv` contains each estimator's release frame, absolute gap, threshold, review status, both candidate lists, and manual-override source. The original V1/V2 directories and manual review checkpoint remain unchanged.

### Suggested review workflow

Use `v3_reconciliation_audit.csv` to inspect the highest-disagreement and missing-candidate rows first. For visual inspection use the existing `scripts/review_basketevent_auto.py` on a **fresh** V1 or V2 auto directory with a new checkpoint filename; it is designed for those existing auto outputs, **not** the new V3 directory. Do not use `v3_review_manifest.csv` as its `--auto-dir` input. Preserve manual-review status and release-frame corrections, then rerun reconciliation into a *new* V3 output directory with the updated review CSV.

## Validity and evaluation protocol

- Agreement is not accuracy: correlated mistakes can cause both estimators to agree on the wrong release.
- The 8 development reviewed shots influenced V3 design. Report any analysis on those shots as development diagnostics only.
- Sample previously unseen shots from **each** of the 4 games before examining their manual release frames. Keep the V1/V2 predictions frozen. Evaluate MAE, median absolute error, within-3-frame rate, and review/abstention coverage separately for agreement and disagreement groups; also report false-agreement rate (both estimators agree within five frames but both miss the manual release by >3 frames).
- Stronger generalization claims require additional held-out games, because within-game images and plays are correlated.
- The release-frame labeler may inspect later frames during *annotation*. H1 model inputs must only use frames up to verified release. Do not expose `actionType`, shot result, event outcome, or later trajectory to predictive features.
