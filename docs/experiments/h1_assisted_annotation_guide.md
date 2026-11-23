# H1 BasketEvent-assisted annotation — manual verification protocol

Use this only with the generated `artifacts/h1_multigame_video_v1/manifest.csv` and its sibling `event_matching_report.csv`. Source manifest and experiment outputs are preserved; the tool writes a separate annotated manifest and checkpoints after each completed shot.

## One-shot smoke test (Windows PowerShell, repository root)

```powershell
uv run --no-sync pytest -q tests/test_annotate_basketevent_assisted.py
uv run --no-sync python scripts/annotate_basketevent_assisted.py --manifest artifacts/h1_multigame_video_v1/manifest.csv --output artifacts/h1_multigame_video_v1/annotated_smoke.csv --limit 1
```

The UI displays BasketEvent's player boxes as *proposals*. Key commands:

- `A` / `D`: previous / next frame; `J` / `L`: previous / next 10 frames
- `R`: explicitly mark the frame where the ball leaves the shooter's hand
- Click the closest player defending the shot to select its box; `X` clears the defender
- `1` / `2`: left / right shooting hand, explicitly inspected by human
- `S`: save one manually verified release-frame annotation; `K`: skip; `Q`: quit

**Important**: The program starts at the clip's midpoint as a navigation convenience **not** as a guessed release frame. A shot cannot be saved until the user explicitly presses `R` and selects a shooting hand. Verify shooter identity, selected frame, boxes, and defender by looking at the actual video. Saving does not establish that tracking is physically accurate. The defender field can be empty if no defender is identifiable.

For a larger batch, use a **new** output path, e.g. `annotated_batch.csv`. A run creates its output file on first launch, checkpoints changes after each saved shot, and refuses to overwrite existing files. To continue an interrupted run, supply the previously generated `annotated_batch.csv` as `--manifest`, and use another **new** `--output` path. The sibling event report must remain next to the manifest.

## Assumptions and limitations

- Expects BasketEvent trajectories to contain per-frame `[x, y, width, height]` entries in the same pixel coordinate space as the decoded MP4. Aborts a shot when annotated trajectory length differs from decoded frame count; matching lengths alone do not prove alignment.
- Relies on `source_role` from the saved shot-matching audit, never picks a person by event outcome.
- Dataset may include multiple tracks for one real player, ambiguous defender identities, missing or bad bounding boxes, and frame-index alignment errors. Audit with eyes before model evaluation.
- Does not use post-release video as a model feature. Human review may use video to identify the exact release moment; all saved measurement boxes are from that release frame.
- Keep the complete manifest, including failures and skips, for missingness-aware coverage reporting. For modeling, keep game-held-out splits and do not use outcomes to guide annotation or shot selection.
