# BasketEvent Raw Candidate Failure Audit

## Purpose

This exploratory analysis investigates why the frozen release detector does not generate geometrically qualifying release candidates inside certain manually estimated shot-release intervals.

It builds on the human-referenced candidate-filter audit from PR #43. That audit found seven provisional release intervals without overlapping raw candidates, even before the `steady()` shooter-box filter was applied.

## Method

Seven BasketEvent test clips were selected because their provisional human release intervals contained no raw detector candidates.

The diagnostic script examines the original shooter and ball trajectories using the frozen detector's criteria:

- Valid shooter bounding box
- Valid ball bounding box
- Ball position within the specified hand-reach region
- At least three valid future ball positions
- Sufficient upward ball displacement over the future-frame window

The detector uses `HAND_REACH = 0.25`, `RISE_FRAMES = 7`, and `RISE_BH = 0.3`.

The script examines both the exact provisional interval and an additional three frames on either side. This wider window is diagnostic only and does not redefine the human-estimated release interval.

Each frame is assigned its first failing condition in evaluation order. A reported reason does not exclude the possibility that other conditions also fail.

No frozen detector logic, thresholds, model predictions, or trajectory data were changed.

## Results

| Annotated shot | Human interval | Exact-frame observations | Qualifying candidates within ±3-frame window |
|---|---|---|---:|
| Sengun | 84–85 | Rise threshold not met at either frame | 1 |
| Anderson | 121–122 | Outside hand-reach region at both frames | 1 |
| McBride | 143–144 | Outside hand-reach region at both frames | 2 |
| Dunn | 124–125 | Hand-reach condition fails, then rise threshold fails | 0 |
| Johnson — layup | 85–86 | Missing ball boxes at both frames | 0 |
| Bailey | 57–58 | Missing ball boxes at both frames | 0 |
| Nance Jr. | 87–88 | Rise threshold fails, then hand-reach condition fails | 0 |

### Nearby candidates outside the provisional interval

Sengun, Anderson, and McBride had qualifying raw candidates within three frames of their provisional intervals.

This may reflect small timing differences between the manual observations and the geometric detector conditions. The analysis cannot establish which frame is physically correct.

### Missing ball trajectories

Johnson's layup and Bailey's three-pointer had no valid ball bounding boxes at either provisional release frame.

This prevents those frames from satisfying the raw candidate criteria. It does not independently establish why the tracking data is missing.

### Unusual shot mechanics

Dunn's alley-oop and Nance Jr.'s blocked layup failed different geometric requirements.

The provisional Dunn release is particularly uncertain because ball-hand separation was difficult to establish. The Nance Jr. observation may represent defender-induced separation rather than a conventional shooter-initiated release.

These cases should not be treated as ordinary verified releases.

## Reproduction

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m tools.audit_release_raw_candidate_failures `
    --audit artifacts/release_candidate_filter_audit/per_clip.csv `
    --json-root data/basketevent `
    --output artifacts/release_raw_candidate_failure_audit/per_clip.csv `
    --radius 3
```

The local per-clip output from PR #43 and corresponding BasketEvent trajectory JSON files are required.

## Validation

- 11 new tests passed.
- Full test suite: 384 passed, 2 skipped.
- Ruff checks passed.
- The diagnostic condition reconstruction agreed with the frozen candidate-generation function across the inspected frames.

## Limitations

- Only seven clips were examined in this follow-up audit.
- Human release intervals remain provisional and were not independently adjudicated.
- Some visual interpretations were AI-assisted.
- Failure reasons represent algorithmic conditions, not established physical causes.
- The first-failing-condition classification does not capture every condition that might fail.
- Missing tracked boxes cannot independently be attributed to a specific tracking-system defect.
- Nearby qualifying candidates are not necessarily true release frames.
- Different shot mechanics and occlusions may complicate interpretation.
- No frozen H1/H2 evaluation or detector threshold was changed.

## Conclusion

The seven clips without raw candidate overlap showed several distinct failure patterns.

Some clips contained nearby qualifying candidates outside the provisional interval, two lacked valid ball boxes at the observed frames, and others failed hand-reach or rising-ball conditions.

These findings clarify the frozen detector's behavior but do not establish that modifying candidate criteria would improve release accuracy or downstream shot prediction.