# Blind release audit

The completed local audit is in `reports/release_audit_50/report.md`. The working
audit, original videos, full-resolution decoded frames, and source labels remain
under `data/release_audit_50` and `data/release_audit_source`. `data/` is gitignored;
preserve those directories separately if moving machines. Nothing was committed
or pushed, and no teammates were contacted.

The detector is unchanged from `feature/basketevent-h1` commit
`8a45ea090e5b1bfaae50ceca1460076a5e81305c`. The audit implementation lives on
`audit/blind-release-50`. Sampling refuses a modified detector, duplicate IDs,
an unexpected population count, or an existing output directory.

## Population and acquisition

BasketEvent revision `de20e7ec434d6cb0e79e75bdb4de47da84e57c0f` contains 1,941
video-associated label files, with 851 unique made/missed shot events across
11 games. Live NBA play-by-play returned 403 responses or timed out. The pool
builder therefore matched the cached official shot chart using game, period,
and integer-second clock, resolving ties by player surname and then outcome.
Outcome was used only for event linkage and never for selection.

850 events have unambiguous official shot IDs. One has two indistinguishable
official candidates (650 and 724); it remains eligible with a source-derived ID.
It was not sampled. The selected 50 all have official IDs. The pool discrepancy
was documented before the sample freeze. All 50 original clips decoded at 25 FPS
with frame counts equal to their tracking trajectories. Later track overlays at
reference frames were visually aligned with the decoded actors; no gross frame
offset was observed. Equal counts alone are not a frame-alignment proof.

`scripts/acquire_release_audit.py` retrieves the pinned inventory and paired
labels. `scripts/build_release_audit_pool.py` rebuilds the pool from those labels
and `data/raw/nba_2025_26/source_shots.csv`. That cached table's SHA-256 is
`49ceb967ff53f0e6140d9824b8c3d619f30bb363a1e12efe056d11e9a727955c`.

## Reproduction

Run from the repository root with Python 3.11, the locked project dependencies,
Pillow, certifi, and FFmpeg/FFprobe available. The actual run used a separate
environment at `/private/tmp/nba-audit-env` plus locked `nba-api==1.11.6` for the
existing detector module's import. No training or chronological test evaluation
was performed.

```sh
export PYTHONPATH=src
python scripts/build_release_audit_pool.py
python -m nba_shots.cli sample-release-audit \
  --pool data/release_audit_source/pool.csv \
  --output data/new_blind_release_audit --n 50 --seed 42 \
  --expected-population 851 --download
python -m nba_shots.cli release-review-frames \
  --audit data/new_blind_release_audit --shot-id SHOT_ID --step 25
```

Sampling sorts shot IDs, then calls Python's `random.Random(42).sample` without
replacement. Only IDs and clip locations appear in the blind manifest. Original
predictions, selected entity keys, source revision, and input hashes are frozen
in the private package. Inspect only `blind/` during reference annotation.

Review uniformly spaced contact sheets, then choose detailed windows visually.
The existing `scripts/release_audit_details.py` accepts manually chosen windows
and crops for this audit and logs them. Full-resolution original PNGs are saved
alongside the contact sheets. Official player/period/clock cues identify the
intended attempt in clips containing several attempts; they contain no automatic
frames, chosen track IDs, outcomes, or detection statuses.
Original broadcast scorebugs and visible shot results remained in the decoded
video frames; outcome labels were excluded from the blind manifest.

Complete `blind/references.json` for every selection. Frame indices are zero-based;
the release is the first frame with the ball separated from the shooting hand.
Reference boxes use `[left, top, right, bottom]` original-image pixels. Record
uncertainty and invisible opponents explicitly. `identity_frame` anchors a clear
actor reference when release timing itself is uncertain. Never replace a selected
failure. Freeze before opening private predictions:

```sh
python -m nba_shots.cli freeze-release-references --audit AUDIT_DIRECTORY
python -m nba_shots.cli unblind-release-audit --audit AUDIT_DIRECTORY
# Inspect overlays; complete unblinded/identity_reviews.json with visual judgments.
python -m nba_shots.cli score-release-audit \
  --audit AUDIT_DIRECTORY --output NEW_REPORT_DIRECTORY
```

For the completed audit, reproduce scoring and its evidence bundle in a new directory:

```sh
python scripts/package_release_audit.py \
  --audit data/release_audit_50 --output reports/release_audit_50_reproduced
python -m pytest -q
```

The package command checks deterministic selection, every frozen clip/label hash,
all 50 comparisons, and the timing denominators. It copies frozen inputs and
references and creates reference-frame overlays plus representative side-by-side
timing failures. SHA-256 provenance permits comparison with the original run.

## Interpretation

Timing accuracy uses clear aligned release references; a missing prediction is a
failure. The inclusive tolerance is ±3 frames (±0.12 seconds at 25 FPS). Identity
accuracy has its own role-specific denominator and uses visual actor agreement
at the reference frame, retaining the tracks originally selected at the automatic
frame. Approximate box overlap supports review but does not decide agreement.

These are single-assistant visual references, without independent human
adjudication or inter-reviewer agreement. Close feet distances, occlusion,
compression, and one-frame release ambiguity can affect the references. Results
apply to this video-available subset; they do not establish general NBA accuracy
or the predictive H1 result. Any detector tuning after this audit needs a fresh
blind sample for an independent accuracy claim.
