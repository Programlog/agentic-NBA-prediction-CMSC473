"""Select BasketEvent JSON shot events, match NBA shots, optionally fetch MP4s.

No shot outcome or later video frames are supplied to CV features. BasketEvent
video/trajectory contents must be inspected before establishing release frames.
All output files are new; this script refuses to overwrite an existing run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = "zaywas/BasketEvent"
CLOCK_RE = re.compile(r"^PT(\d+)M(\d+(?:\.\d+)?)S$")
SHOT_LABELS = {"made shot", "missed shot"}


def stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def parse_clock(value: object) -> float:
    match = CLOCK_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"Invalid NBA period clock: {value}")
    return int(match.group(1)) * 60 + float(match.group(2))


def indexed_paths(files: list[str]) -> dict[str, list[str]]:
    """Return JSON files grouped by game, limited to filenames with MP4 pairs."""
    known = set(files)
    games = defaultdict(list)
    for name in files:
        parts = name.split("/")
        if (
            len(parts) == 3
            and parts[0] in {"train", "valid", "test"}
            and parts[1].isdigit()
            and name.lower().endswith(".json")
            and name[:-5] + ".mp4" in known
        ):
            games[parts[1]].append(name)
    return {game: sorted(paths) for game, paths in games.items()}


def extract_events(data: object, json_path: str) -> list[dict]:
    """Collect shot annotations (possibly multiple per clip) without choosing outcomes."""
    if not isinstance(data, dict):
        raise ValueError("Annotation root must be a JSON object")
    seen = set()
    records = []
    for role, annotation in data.items():
        if not isinstance(annotation, dict):
            continue
        event = annotation.get("event")
        if not isinstance(event, dict):
            continue
        label = str(event.get("actionType", "")).strip().lower()
        if label not in SHOT_LABELS:
            continue
        signature = (
            str(event.get("clock")),
            str(event.get("period")),
            str(event.get("playerName")),
            str(event.get("description")),
            label,
        )
        if signature in seen:
            continue
        seen.add(signature)
        records.append(
            {
                "json_path": json_path,
                "video_path": json_path[:-5] + ".mp4",
                "source_role": role,
                "event_action": event.get("actionType"),
                "event_clock": event.get("clock"),
                "event_period": event.get("period"),
                "event_player": event.get("playerName"),
                "event_description": event.get("description"),
                "event_team": event.get("teamTricode"),
                "event_type": label,
            }
        )
    return records


def nba_matches(event: dict, game: str, nba: pd.DataFrame) -> tuple[str, dict | None]:
    """Require unique game + period + time; NEVER use outcome to disambiguate."""
    try:
        seconds = parse_clock(event["event_clock"])
        period = int(event["event_period"])
    except (ValueError, TypeError) as exc:
        return f"invalid_event_time: {exc}", None
    frame = nba.loc[
        nba["game_id"].eq(game)
        & nba["period"].eq(period)
        & (nba["game_clock_seconds"] - seconds).abs().lt(0.01)
    ]
    if len(frame) == 0:
        return "no_nba_match", None
    if len(frame) > 1:
        return "ambiguous_nba_match", None
    shot = frame.iloc[0]
    # Outcome is only an audit check AFTER the unique match, not a match key.
    expected = 1 if event["event_type"] == "made shot" else 0
    if int(shot["y_true"]) != expected:
        return "outcome_mismatch", None
    return "matched", {
        "shot_id": str(shot["shot_id"]),
        "game_id": str(shot["game_id"]),
        "game_event_id": str(shot["game_event_id"]),
    }


def choose_games(groups: dict[str, list[str]], games: int, seed: int) -> list[str]:
    if games < 1:
        raise ValueError("--games must be >= 1")
    return sorted(groups, key=lambda game: stable_order(seed, game))[:games]


def filter_games_with_nba_coverage(
    groups: dict[str, list[str]],
    nba_game_ids: set[str],
) -> dict[str, list[str]]:
    """Keep only BasketEvent games present in the NBA shot source."""
    return {game: paths for game, paths in groups.items() if game in nba_game_ids}


def build(
    *,
    shots: Path,
    output: Path,
    games: int = 5,
    shots_per_game: int = 10,
    max_json_per_game: int = 40,
    seed: int = 42,
    download_videos: bool = False,
    repo: str = REPO,
) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    shots, output = Path(shots), Path(output)
    if games < 1 or shots_per_game < 1 or max_json_per_game < 1:
        raise ValueError("games, shots_per_game, max_json_per_game must be positive")
    if output.exists():
        raise ValueError(f"Output path already exists: {output}; use a new one")
    if not shots.is_file():
        raise FileNotFoundError(shots)
    nba = pd.read_parquet(shots)
    required = {"shot_id", "game_id", "game_event_id", "period", "game_clock_seconds", "y_true"}
    if missing := required - set(nba):
        raise ValueError(f"NBA shot table missing: {sorted(missing)}")
    if nba.shot_id.isna().any() or nba.shot_id.duplicated().any():
        raise ValueError("NBA shot IDs must be unique and nonmissing")
    nba = nba.copy()
    nba["game_id"] = nba.game_id.astype(str).str.zfill(10)
    nba["period"] = pd.to_numeric(nba.period, errors="raise")
    nba["game_clock_seconds"] = pd.to_numeric(nba.game_clock_seconds, errors="raise")
    api = HfApi()
    # Pin a Hub revision (when supplied) for an externally reproducible dataset snapshot.
    info = api.repo_info(repo_id=repo, repo_type="dataset")
    revision = info.sha
    all_paths = api.list_repo_files(repo_id=repo, repo_type="dataset", revision=revision)
    grouped = indexed_paths(all_paths)
    if not grouped:
        raise ValueError("No paired JSON+MP4 game files found")

    nba_game_ids = set(nba["game_id"])
    covered_groups = filter_games_with_nba_coverage(grouped, nba_game_ids)

    if not covered_groups:
        raise ValueError(
            "No BasketEvent game folders overlap with the NBA shot table. "
            "Check the source season and game IDs before downloading JSONs."
        )

    chosen = choose_games(covered_groups, games, seed)
    output.mkdir(parents=True)
    json_audit = []
    events = []
    local_root = Path("data/basketevent")
    for game in chosen:
        candidates = sorted(grouped[game], key=lambda p: stable_order(seed, p))[:max_json_per_game]
        for index, name in enumerate(candidates, 1):
            row = {
                "game_id": game,
                "json_path": name,
                "status": "error",
                "error": "",
                "shot_event_count": 0,
            }
            try:
                local = hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=name,
                    revision=revision,
                    local_dir=local_root,
                )
                data = json.loads(Path(local).read_text(encoding="utf-8"))
                found = extract_events(data, name)
                for item in found:
                    item["game_id"] = game
                events.extend(found)
                row.update({"status": "ok", "shot_event_count": len(found)})
            except Exception as exc:
                row["error"] = str(exc)
            json_audit.append(row)
            print(
                f"[{game} {index}/{len(candidates)}] {row['status']} | "
                f"shot annotations={row['shot_event_count']}",
                flush=True,
            )
    # Record EVERY extracted candidate event, including unmatched and duplicates.
    event_report = []
    seen_shots = set()
    eligible = defaultdict(list)
    for event in events:
        status, nba_shot = nba_matches(event, event["game_id"], nba)
        record = dict(event)
        record.update(
            {
                "match_status": status,
                "shot_id": "",
                "game_event_id": "",
                "selection_status": "not_eligible",
                "download_status": "not_requested",
            }
        )
        if nba_shot is not None:
            record.update(nba_shot)
            if nba_shot["shot_id"] in seen_shots:
                record["match_status"] = "duplicate_matched_shot"
            else:
                seen_shots.add(nba_shot["shot_id"])
                eligible[event["game_id"]].append(record)
        event_report.append(record)
    for game in chosen:
        entries = sorted(eligible[game], key=lambda r: stable_order(seed, r["shot_id"]))
        for index, record in enumerate(entries):
            record["selection_status"] = (
                "selected" if index < shots_per_game else "not_selected_quota"
            )
    selected = [r for r in event_report if r["selection_status"] == "selected"]
    for record in selected:
        if not download_videos:
            continue
        try:
            video = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=record["video_path"],
                revision=revision,
                local_dir=local_root,
            )
            record["download_status"] = "ok"
            record["local_clip"] = os.path.relpath(Path(video).resolve(), output.resolve())
        except Exception as exc:
            record["download_status"] = "error"
            record["download_error"] = str(exc)
    manifest = []
    for record in selected:
        manifest.append(
            {
                "shot_id": record["shot_id"],
                "game_id": record["game_id"],
                "game_event_id": record["game_event_id"],
                "clip_path": record.get("local_clip", ""),
                "release_frame": "",
                "shooter_box": "",
                "defender_box": "",
                "shooting_side": "",
                "source_json": record["json_path"],
                "download_status": record["download_status"],
                "annotation_status": "pending_manual_review",
            }
        )
    manifest_cols = [
        "shot_id",
        "game_id",
        "game_event_id",
        "clip_path",
        "release_frame",
        "shooter_box",
        "defender_box",
        "shooting_side",
        "source_json",
        "download_status",
        "annotation_status",
    ]
    pd.DataFrame(json_audit).to_csv(output / "json_audit.csv", index=False)
    pd.DataFrame(
        event_report,
        columns=[
            "game_id",
            "json_path",
            "video_path",
            "source_role",
            "event_action",
            "event_clock",
            "event_period",
            "event_player",
            "event_description",
            "event_team",
            "event_type",
            "match_status",
            "shot_id",
            "game_event_id",
            "selection_status",
            "download_status",
            "download_error",
            "local_clip",
        ],
    ).to_csv(output / "event_matching_report.csv", index=False)
    pd.DataFrame(manifest, columns=manifest_cols).to_csv(output / "manifest.csv", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repo,
        "dataset_revision": revision,
        "basket_event_games_with_paired_files": len(grouped),
        "games_with_nba_coverage": len(covered_groups),
        "games_excluded_no_nba_coverage": len(grouped) - len(covered_groups),
        "nba_source": str(shots),
        "nba_source_sha256": hashlib.sha256(shots.read_bytes()).hexdigest(),
        "seed": seed,
        "games_requested": games,
        "games_selected": chosen,
        "max_json_per_game": max_json_per_game,
        "shots_per_game": shots_per_game,
        "json_attempted": len(json_audit),
        "json_failed": sum(r["status"] != "ok" for r in json_audit),
        "shot_event_annotations": len(events),
        "match_status_counts": dict(Counter(r["match_status"] for r in event_report)),
        "selected_unique_shots": len(selected),
        "video_downloads_requested": bool(download_videos),
        "video_download_succeeded": sum(r["download_status"] == "ok" for r in selected),
        "video_download_failed": sum(r["download_status"] == "error" for r in selected),
        "note": (
            "Shots are selected by deterministic hash, never made/missed status. "
            "Event outcomes are checked only after an unambiguous ID match. "
            "No release frames or player xyxy boxes are inferred from "
            "BasketEvent trajectories."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=Path, default=Path("data/recent/shots.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--shots-per-game", type=int, default=10)
    parser.add_argument("--max-json-per-game", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download MP4s for selected matched shots (off by default)",
    )
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
