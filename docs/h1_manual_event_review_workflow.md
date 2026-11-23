# H1 Commit 6: Independent manual event-review worksheet

## Scope

This commit adds an offline, non-destructive workflow for preparing and checking
human review observations for the existing eight BasketEvent video candidates.
It does **not** independently view videos, establish shot identity, validate
release frames, or approve training data.

The workflow deliberately separates the previous *provisional* release-frame
intervals from an independent reviewer's observed timing fields. It never copies
the proposed release into an observed release field.

## Workflow

1. Use `prepare` to create a fresh worksheet from the ignored local
   `visual_verification_v2.csv` report.
2. A human reviewer opens the actual clip and independently inspects player
   number/team, period, visible clock, action type, and ball release. The
   reviewer records evidence and unresolved discrepancies. It is acceptable
   to leave the decision `uncertain` or `not_reviewed`.
3. Use `validate` to reject missing evidence, malformed frame intervals,
   duplicate/missing shots, and edits to immutable source-context columns.
4. Any positive identity/timing finding remains
   `human_claim_pending_independent_adjudication`; it is **never** training
   approved. A second independent review and explicit adjudication are future
   work, not silently implied by this tool.

**Do not use the actual shot outcome (make/miss) to choose pre-release
frames or engineer pre-release features.**

## Local commands (from repository root)

```powershell
uv run --frozen python scripts/prepare_h1_manual_review.py prepare `
  --source artifacts/h1_video_verification_8clips_v1/visual_verification_v2.csv `
  --output artifacts/h1_video_verification_8clips_v1/independent_review_worksheet_v1.csv

uv run --frozen python scripts/prepare_h1_manual_review.py validate `
  --source artifacts/h1_video_verification_8clips_v1/visual_verification_v2.csv `
  --worksheet artifacts/h1_video_verification_8clips_v1/independent_review_worksheet_v1.csv `
  --output artifacts/h1_video_verification_8clips_v1/independent_review_validation_v1.json

uv run --frozen python -m pytest tests/test_prepare_h1_manual_review.py -q
```

Outputs refuse to overwrite existing paths. Original reviews, download
reports, manifests, video files, and prior Commit 4/5 audit outputs are unchanged.

## Limitations

The script validates completeness and consistency of human-entered records, not
whether the human observation is true. It does not inspect/decode video, test
synchronization, confirm independent reviewer identity, or support adjudicated
training approvals. The historical eight-clip pilot still has zero conclusively
verified events, zero exact verified releases, and zero approved training clips.

The source CSV and generated worksheet are local experiment artifacts, not
tracked inputs; reproducing the pilot on a fresh clone requires those local
files. Synthetic unit tests can run without the videos, internet, or GPU.
