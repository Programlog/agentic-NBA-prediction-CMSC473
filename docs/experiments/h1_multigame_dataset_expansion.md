# H1 Experiment: BasketEvent Multi-Game Dataset Expansion

**Status:** The five-game metadata audit and revised whole-second matching run are complete. Both processed 200 JSON files containing 101 recognized shot events. The original exact-clock rule matched 93 events, while the revised production script matched all 101 with no unmatched or ambiguous events. Forty shots were selected across four contributing games. Automated tests passed; video verification, release annotations, and predictive evaluation remain pending.

## Research question

Can we reproducibly identify and match BasketEvent shot-event clips to existing NBA shot records across several games, preserve failed and ambiguous matches, and build a release-annotation manifest without downloading all videos?

## Motivation and prior observations

- A local inventory listed **3,281 BasketEvent MP4s across 24 game folders**. This is not a claim that all files are NBA shot clips or can be matched.
- The existing H1 motion pilot contains **10 annotated shots from one game** (`0022500003`), insufficient for matched held-out multi-game training and evaluation.
- A 15-JSON-file exploration of game `0022500003` observed event annotations: **Made Shot 6, Missed Shot 4, Rebound 5, Foul 2, ast 1, Free Throw 1**. These are *event annotations*, not a verified count of unique shots.
- BasketEvent's documented player trajectories use **xywh bounding boxes**. We must **not** treat them as the release-frame `xyxy` annotations expected by our H1 scripts. The new manifest intentionally leaves release frame and boxes blank. [BasketEvent authors' repository](https://github.com/zhangyu2003/BasketEvent)

## Protocol before running

- Identify BasketEvent game folders with paired MP4/JSON files and check which game IDs are also present in the supplied NBA shot table.
- Select up to 5 game folders with NBA shot-data coverage using deterministic SHA-256 ordering (seed 42). Record the total number of BasketEvent games, how many overlap with the NBA shot table, and how many are excluded due to missing NBA game coverage. If no games overlap, stop with a clear error before downloading annotations.
- Select up to 40 JSON annotation files per selected game using a deterministic hash of dataset paths, independent of video outcomes.
- Count shot-event annotations (`Made Shot` and `Missed Shot`), deduplicate exact repeated annotations within each clip, then match to the NBA shot table by **game ID, period, and normalized whole-second game clock**. The BasketEvent clock may include fractional seconds, while the NBA shot table stores integer seconds; use the integer-second portion of the BasketEvent clock. Reject ambiguous matches rather than choosing the nearest event or using its made/missed outcome.
- Never use the made/missed label to resolve an ambiguous NBA match. After a unique match, check whether its outcome agrees with the annotation; disagreements become outcome_mismatch audit entries, not selected examples.
- Deduplicate NBA shot IDs across clips; select up to 10 uniquely matched shots per game by deterministic hash.
- Keep failed JSON downloads, missing/ambiguous NBA matches, mismatched outcomes, duplicate matches, quota exclusions, and failed MP4 downloads in CSV reports.
- Stage 1: JSON-only metadata audit; Stage 2: download selected videos only after reviewing candidate coverage and match quality.
- No automatic release-frame inference. Manually review/annotate each selected clip before CoTracker3/YOLO extraction. Use only pre-release visual features, never the event outcome or post-release video.
- Important limitation: Game-level coverage does not guarantee that individual BasketEvent clips match NBA shot records. Event-level matching and outcome consistency must still be verified before shots are accepted for the H1 experiment.

## Command: JSON-only dry pilot (no video downloads)

From the repository root in PowerShell, ensure that the NBA shot dataset is available at `data/processed/nba_2025_26/shots.parquet`, or replace `--shots` with the path to your locally prepared dataset.

The NBA dataset is not included in Git. Each developer must provide it locally. The command below assumes the file exists in the current worktree.

```powershell
uv run --frozen --with huggingface-hub==2.4.0 python scripts/build_basketevent_multigame.py `
  --shots data/processed/nba_2025_26/shots.parquet `
  --output artifacts/h1_matching_5games_wholesecond_v1 `
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

Inspect the matching results and coverage counts before deciding whether to download MP4s. The script now selects only BasketEvent games that appear in the supplied NBA shot table. However, game-level coverage does not guarantee valid shot-level matches. If no games overlap, the script stops before downloading annotations. If selected games produce few or no valid event matches, investigate the source data and matching rules rather than silently substituting outcomes or changing the registered selection.

## Command: separate video run (new output directory)

```powershell
uv run --frozen --with huggingface-hub==2.4.0 python scripts/build_basketevent_multigame.py `
  --shots data/processed/nba_2025_26/shots.parquet `
  --output artifacts/h1_matching_5games_videos_v1 `
  --games 5 `
  --max-json-per-game 40 `
  --shots-per-game 10 `
  --seed 42 `
  --download-videos
```

This repeats the exact selection, reuses downloaded files from cache, and downloads only MP4s for selected uniquely matched shots. The `manifest.csv` clip paths are relative to the manifest directory, compatible with the existing H1 tools. The generated files and clips stay local by default (under ignored `artifacts/` and `data/`).

## Interpreting results / limitations

- The two-game and five-game JSON-only metadata pilots have been executed and their summary files inspected. These results establish automated metadata-matching feasibility, not verified video-to-shot identity or predictive model performance.
- Only JSONs that have a paired MP4 path in the Hub index are considered.
- Whole-second clock normalization addresses fractional-second differences observed in the pilot, but may still produce ambiguous or incorrect matches. Ambiguous matches must remain unresolved, and footage identity requires independent verification.
- BasketEvent event annotations may be noisy and may include duplicate event records or incorrect participant labels.
- Sampling limited numbers of JSONs per game is a pipeline feasibility pilot, **not** representative whole-season evaluation.
- The NBA shot dataset must contain the corresponding BasketEvent games. For this pilot, the dataset contained 102,400 shot records across 573 games and was stored at `data/processed/nba_2025_26/shots.parquet` in a separate local worktree. The dataset is not included in Git, so its location may differ between environments. Unmatched events do not necessarily indicate incorrect BasketEvent annotations; differences in game coverage, timestamp precision, and event metadata must be investigated.
- The code does not infer shooting pose, player identity or exact release timing. These require independent annotation and audits.
- **Predictive evaluation is pending** until enough independently annotated, multi-game shots exist and game-disjoint train/validation/test splits can be frozen.

## Exploratory clock-matching diagnostic

The five-game metadata pilot (artifacts/h1_matching_5games_40json_v1) processed 200 BasketEvent JSON files and identified 101 shot-event annotations. Under the original exact-clock rule, 93 matched NBA shot records and 8 had no match.

A read-only investigation of the eight unmatched events found that each had exactly one NBA shot candidate within two seconds, with the NBA clock equal to the integer-second portion of the BasketEvent clock.

Applying that integer-second rule to all 101 events produced:
- 101 unique clock matches
- 0 ambiguous matches
- 0 unmatched events
- 0 outcome disagreements
- 0 duplicate matched NBA shot IDs

The NBA source contained 102,400 uniquely identified shots across 573 games, dated October 21, 2025 through January 29, 2026. All required matching columns were present, but defender distance, shot clock, dribbles, and touch time were entirely missing.

These results support whole-second clock normalization for this dataset, but they do not independently verify footage identity or prove the rule works across every game.

The revised whole-second matching rule was subsequently validated using the production script in a separate output directory: `artifacts/h1_matching_5games_wholesecond_v1`. The rerun processed the same 200 JSON files and 101 shot-event annotations, producing 101 matched events, 0 unmatched events, and 0 ambiguous matches. The NBA source SHA-256 and BasketEvent revision were unchanged.

The result confirms that the revised implementation reproduces the exploratory clock-matching diagnostic on this cohort. It does not independently establish video-to-shot identity or prove that the rule generalizes to every BasketEvent game.

### Complete annotation-coverage audit for game `0022500001`

The original five-game metadata experiment selected game `0022500001`, but its first 40 JSON files contained no recognized shot-event annotations.

A subsequent exploratory investigation examined an additional 40 JSON files and again found no event fields. To determine whether the missing event metadata extended across the game's available annotations, a separate audit downloaded and inspected all 300 paired JSON files for game 0022500001 in the pinned BasketEvent revision, without downloading any MP4 videos.

The complete audit produced:

- 300 JSON files processed
- 0 retrieval or parsing failures
- 0 files containing an `event` field
- 0 files containing non-null event metadata
- 0 files containing recognized shot events
- 0 recognized made/missed shot annotations

The results are stored locally in `artifacts/h1_game0001_coverage_audit_v1`, including `annotation_coverage.csv` and `summary.json`.

This establishes that none of the 300 examined JSON annotation files contains usable event metadata under the current schema. It does not establish whether the corresponding videos visually contain basketball shots.

Game `0022500001` remains part of the original five-game sampling denominator and contributes zero metadata-matched shots. It must not be silently replaced, and NBA shot outcomes must not be used to fabricate missing BasketEvent annotations.

This limitation demonstrates why NBA game-ID coverage and paired video availability are insufficient measures of actual shot-annotation coverage. Other games require separate coverage audits before conclusions are drawn about the entire BasketEvent dataset.

## Results and reproducibility

### Completed JSON-only pilots

| Metric | Two-game smoke test | Five-game exact-clock audit | Five-game whole-second audit |
|---|---:|---:|---:|
| Selected game folders | 2 | 5 | 5 |
| JSON files attempted | 10 | 200 | 200 |
| JSON failures | 0 | 0 | 0 |
| Shot-event annotations | 6 | 101 | 101 |
| NBA matches | 6 | 93 | 101 |
| Unmatched events | 0 | 8 | 0 |
| Ambiguous matches | 0 | 0 | 0 |
| Selected unique shots | 6 | 40 | 40 |
| Games contributing selected shots | 2 | 4 | 4 |
| MP4 videos downloaded | 0 | 0 | 0 |

Both runs used the BasketEvent dataset revision `c8629460c68e6ba65375d2c4f6d8638650b23117` and the same NBA source hash:

`7d2c17779e65a4c70b334b796133e65f21e97d444e3b01debbde7f9936bb540f`

The five-game audit used seed 42, up to 40 JSON files per game, and a maximum of 10 selected shots per game.

The exploratory whole-second diagnostic initially identified 101 unique NBA matches. A subsequent run of the updated production script independently reproduced 101 matches on the same 200 JSON files. The revised script's summary reported no unmatched, ambiguous, or outcome-mismatched events.

### Pending reproducibility checks

For subsequent video annotation and predictive experiments, record:

- Git commit and Python environment versions
- Revised experiment output directory
- Dataset revision and NBA source hash
- JSON attempts, failures, and match-status counts
- Per-game shot-event and selected-shot counts
- Changes from the original exact-clock pilot
- Video-download outcomes and manual verification results
- Release-annotation and CV feature coverage
- Remaining limitations and implications for predictive evaluation