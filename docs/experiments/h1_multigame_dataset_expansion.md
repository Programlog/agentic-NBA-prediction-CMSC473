# H1 Experiment: BasketEvent Multi-Game Dataset Expansion

**Status:** implementation ready; real multi-game run not yet executed. The counts below are *prior observations*, not outcomes from the new script.

## Research question

Can we reproducibly identify and match BasketEvent shot-event clips to existing NBA shot records across several games, preserve failed and ambiguous matches, and build a release-annotation manifest without downloading all videos?

## Motivation and prior observations

- A local inventory listed **3,281 BasketEvent MP4s across 24 game folders**. This is not a claim that all files are NBA shot clips or can be matched.
- The existing H1 motion pilot contains **10 annotated shots from one game** (`0022500003`), insufficient for matched held-out multi-game training and evaluation.
- A 15-JSON-file exploration of game `0022500003` observed event annotations: **Made Shot 6, Missed Shot 4, Rebound 5, Foul 2, ast 1, Free Throw 1**. These are *event annotations*, not a verified count of unique shots.
- BasketEvent's documented player trajectories use **xywh bounding boxes**. We must **not** treat them as the release-frame `xyxy` annotations expected by our H1 scripts. The new manifest intentionally leaves release frame and boxes blank. [BasketEvent authors' repository](https://github.com/zhangyu2003/BasketEvent)

## Protocol before running

- Select **5 game folders** by SHA-256 seeded ordering (seed 42) across available paired MP4/JSON paths.
- Select up to **40 JSON annotation files** per selected game using a deterministic hash of dataset paths, independent of video outcomes.
- Count shot-event annotations (`Made Shot` and `Missed Shot`), deduplicate exact repeated annotations within a clip, then match to NBA shot table by **game ID, period and exact game clock**.
- **Never use the made/missed label to resolve an ambiguous NBA match.** After a *unique* match, check whether its outcome agrees with the annotation; disagreements become `outcome_mismatch` audit entries, not selected examples.
- Deduplicate NBA shot IDs across clips; select up to **10 uniquely matched shots per game** by deterministic hash.
- Keep failed JSON downloads, missing/ambiguous NBA matches, mismatched outcomes, duplicate matches, quota exclusions, and failed MP4 downloads in CSV reports.
- **Stage 1:** JSON-only metadata audit; **Stage 2:** download selected videos only after reviewing candidate coverage and match quality.
- **No automatic release-frame inference.** Manually review/annotate each selected clip before CoTracker3/YOLO extraction. Use only pre-release visual features, never the event outcome or post-release video.

## Command: JSON-only dry pilot (no video downloads)

From repo root in PowerShell:

```powershell
uv run --no-sync python scripts/build_basketevent_multigame.py `
  --shots data/recent/shots.parquet `
  --output artifacts/h1_multigame_metadata_v1 `
  --games 5 `
  --max-json-per-game 40 `
  --shots-per-game 10 `
  --seed 42
```

The script checks that the output directory does **not** already exist. Results:

- `summary.json`: source revision and hash, chosen games, event/match counts, successes/failures
- `json_audit.csv`: every attempted JSON and its download/parse status
- `event_matching_report.csv`: every shot annotation, match status and selection decision
- `manifest.csv`: candidate shots, source JSON, blank release/box annotations

Inspect counts **before** deciding whether to download MP4s. If source NBA data doesn't cover the chosen games, note the no-match results and obtain the corresponding source shots rather than silently substituting outcomes or changing games.

## Command: separate video run (new output directory)

```powershell
uv run --no-sync python scripts/build_basketevent_multigame.py `
  --shots data/recent/shots.parquet `
  --output artifacts/h1_multigame_videos_v1 `
  --games 5 `
  --max-json-per-game 40 `
  --shots-per-game 10 `
  --seed 42 `
  --download-videos
```

This repeats the exact selection, reuses downloaded files from cache, and downloads only MP4s for selected uniquely matched shots. The `manifest.csv` clip paths are relative to the manifest directory, compatible with the existing H1 tools. The generated files and clips stay local by default (under ignored `artifacts/` and `data/`).

## Interpreting results / limitations

- **No result is claimed until a user executes the script** and reports its CSV/JSON outputs.
- Only JSONs that have a paired MP4 path in the Hub index are considered.
- Exact clock matching can miss true events if timestamps differ, and ambiguous matches must remain unresolved.
- BasketEvent event annotations may be noisy and may include duplicate event records or incorrect participant labels.
- Sampling limited numbers of JSONs per game is a pipeline feasibility pilot, **not** representative whole-season evaluation.
- The current `data/recent/shots.parquet` must contain corresponding games; unmatched events cannot be treated as evidence that BasketEvent clips are wrong.
- The code does not infer shooting pose, player identity or exact release timing. These require independent annotation and audits.
- **Predictive evaluation is pending** until enough independently annotated, multi-game shots exist and game-disjoint train/validation/test splits can be frozen.

## Results placeholder (fill after running)

Record Git commit, timestamp, Python and huggingface_hub versions, selected games, Hub dataset revision, NBA table hash, JSON attempts/failures, matched/unmatched/ambiguous/duplicate outcomes, selected clips, downloads, manual annotation outcomes, CV extraction coverage, and observations explaining why next steps changed.
