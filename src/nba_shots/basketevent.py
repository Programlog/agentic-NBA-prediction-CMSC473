"""BasketEvent 2025-26 clips: shot events matched to NBA records and release-time box features.

Each clip's JSON has per-frame [x, y, w, h] boxes for players (with jersey colour) and the
ball, and the shot event (play-by-play V3 fields) attached to the shooter. Release is found
from the boxes alone, so features exist for labeled clips with or without video.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import BOX_FEATURES, environment, utc_now, write_json
from .data import add_player_history, assign_splits
from .vision import normalize_recent

REPO = "zaywas/BasketEvent"
SEASON = "2025-26"
COHORT = "basketevent_2025_26"
FPS = 25.0
SHOT_ACTIONS = {"Made Shot", "Missed Shot"}
HAND_REACH = 0.25  # ball centre within the shooter box widened by a quarter box per side
RISE_BH, RISE_FRAMES = 0.3, 7  # then the ball climbs 0.3 box heights within 7 frames
SPEED_FRAMES, FLIGHT_FRAMES = 5, 4
FEATURE_COLUMNS = sorted(BOX_FEATURES)
EVENT_KEY = ["period", "clock", "description"]


def boxes(entity: dict) -> np.ndarray:
    """Per-frame [x, y, w, h]; missing or empty boxes become NaN rows."""
    raw = entity.get("trajectory") or []
    if isinstance(raw, str):
        raw = json.loads(raw)
    result = np.full((len(raw), 4), np.nan)
    for i, box in enumerate(raw):
        if isinstance(box, list) and len(box) == 4 and None not in box and min(box[2:]) > 0:
            result[i] = box
    return result


def _feet(box: np.ndarray) -> np.ndarray:
    return box[..., :2] + box[..., 2:] * [0.5, 1.0]


def _centre(box: np.ndarray) -> np.ndarray:
    return box[..., :2] + box[..., 2:] / 2


def _at(track: np.ndarray, frame: int) -> np.ndarray:
    return track[frame] if 0 <= frame < len(track) else np.full(4, np.nan)


def _xyxy(box: np.ndarray) -> str | None:
    return (
        json.dumps([round(float(v), 1) for v in [*box[:2], *(box[:2] + box[2:])]])
        if (np.isfinite(box).all())
        else None
    )


def release_frame(shooter: np.ndarray, ball: np.ndarray) -> int | None:
    """Last frame with the ball at the shooter's hands that is followed by the ball climbing."""
    n = min(len(shooter), len(ball))
    x, y, w, h = shooter[:n].T
    bx, by = _centre(ball[:n]).T
    with np.errstate(invalid="ignore"):
        held = (
            (bx > x - HAND_REACH * w)
            & (bx < x + (1 + HAND_REACH) * w)
            & (by > y - HAND_REACH * h)
            & (by < y + h)
        )
    for k in np.flatnonzero(held)[::-1]:
        ahead = by[k + 1 : k + 1 + RISE_FRAMES]
        ahead = ahead[np.isfinite(ahead)]
        if len(ahead) >= 3 and ahead[-1] < by[k] - RISE_BH * h[k]:
            return int(k)
    return None


def clip_features(clip: dict, shooter_key: str) -> dict:
    """Box features at release, in shooter box heights (bh); NaN where not measurable."""
    values = {name: np.nan for name in FEATURE_COLUMNS}
    empty = {**values, "release_frame": None, "shooter_box": None, "defender_box": None}
    shooter, ball = boxes(clip[shooter_key]), boxes(clip.get("ball") or {})
    if not np.isfinite(ball).any():
        return {**empty, "box_status": "no_ball"}
    r = release_frame(shooter, ball)
    if r is None:
        return {**empty, "box_status": "release_not_found"}
    h, feet = shooter[r, 3], _feet(shooter[r])
    values["release_height_bh"] = (feet[1] - _centre(ball[r])[1]) / h
    stance = _feet(shooter[max(0, r - 20) : max(0, r - 9)])[:, 1]  # 0.4-0.8 s before release
    if np.isfinite(stance).any():
        values["jump_bh"] = (np.nanmedian(stance) - feet[1]) / h
    flight = _centre(_at(ball, r + FLIGHT_FRAMES))[1]
    values["ball_rise_bhps"] = (_centre(ball[r])[1] - flight) / h * FPS / FLIGHT_FRAMES

    team = clip[shooter_key].get("jersey_color")
    others = [
        (boxes(v), team is not None and v.get("jersey_color") not in (None, team))
        for k, v in clip.items()
        if k not in (shooter_key, "ball") and isinstance(v, dict)
    ]
    opponents = [
        (track, np.linalg.norm(_feet(_at(track, r)) - feet) / h) for track, o in others if o
    ]
    opponents = [(track, d) for track, d in opponents if np.isfinite(d)]
    defender = None
    if opponents:
        defender, nearest = min(opponents, key=lambda pair: pair[1])
        values["defender_distance_bh"] = nearest
        values["defenders_within_1bh"] = float(sum(d <= 1 for _, d in opponents))
    p = r - SPEED_FRAMES
    if p >= 0:
        if defender is not None:
            earlier = np.linalg.norm(_feet(_at(defender, p)) - _feet(shooter[p])) / h
            values["defender_closing_bhps"] = (earlier - nearest) * FPS / SPEED_FRAMES
        # Subtract everyone else's median motion so camera pans do not count as movement.
        moves = [_feet(_at(track, r)) - _feet(_at(track, p)) for track, _ in others]
        moves = [m for m in moves if np.isfinite(m).all()]
        own = feet - _feet(shooter[p])
        if moves and np.isfinite(own).all():
            relative = own - np.median(moves, axis=0)
            values["shooter_speed_bhps"] = np.linalg.norm(relative) / h * FPS / SPEED_FRAMES
    return {
        **{k: float(v) for k, v in values.items()},
        "release_frame": r,
        "shooter_box": _xyxy(shooter[r]),
        "defender_box": None if defender is None else _xyxy(_at(defender, r)),
        "box_status": "ok",
    }


def clip_rows(path: str, clip: dict) -> list[dict]:
    """One row per distinct shot event in a clip; ambiguous clips keep NaN features."""
    holders: dict[tuple, list[str]] = {}
    for key, entity in clip.items():
        event = entity.get("event") if isinstance(entity, dict) else None
        if key != "ball" and isinstance(event, dict) and event.get("actionType") in SHOT_ACTIONS:
            holders.setdefault(tuple(event.get(f) for f in EVENT_KEY), []).append(key)
    rows = []
    for event_key, keys in sorted(holders.items(), key=str):
        if len(holders) > 1:
            features = {"box_status": "multiple_shot_events"}
        elif len(keys) > 1:
            features = {"box_status": "ambiguous_shooter"}
        else:
            features = clip_features(clip, keys[0])
        action = clip[keys[0]]["event"]["actionType"]
        rows.append({"clip": path, **dict(zip(EVENT_KEY, event_key)), "action": action, **features})
    return rows


def match_events(events: pd.DataFrame, plays: pd.DataFrame) -> pd.Series:
    """NBA event number for each clip event; NaN unless exactly one field goal matches."""
    plays = plays.loc[plays.isFieldGoal.eq(1), ["actionNumber", *EVENT_KEY]].copy()
    plays["description"] = plays.description.str.strip()
    unique = plays.drop_duplicates(EVENT_KEY, keep=False)
    keys = events[EVENT_KEY].assign(
        period=pd.to_numeric(events.period, errors="coerce"),
        description=events.description.astype(str).str.strip(),
    )
    return keys.merge(unique, on=EVENT_KEY, how="left").actionNumber.set_axis(events.index)


def game_rows(clips: dict[str, dict], plays: pd.DataFrame) -> pd.DataFrame:
    columns = ["clip", *EVENT_KEY, "action", "box_status", "release_frame", *FEATURE_COLUMNS]
    rows = [row for path, clip in sorted(clips.items()) for row in clip_rows(path, clip)]
    frame = pd.DataFrame(rows).reindex(columns=columns + ["shooter_box", "defender_box"])
    frame["game_event_id"] = match_events(frame, plays) if len(frame) else []
    # A shot shown in several clips keeps one row, preferring a clip where release was found.
    frame["duplicate_clips"] = frame.groupby("game_event_id")["clip"].transform("size") - 1
    frame = frame.sort_values(["game_event_id", "box_status", "clip"], key=_ok_first)
    keep = frame.game_event_id.isna() | ~frame.duplicated("game_event_id")
    return frame.loc[keep].sort_values("clip").reset_index(drop=True)


def _ok_first(column: pd.Series) -> pd.Series:
    return column.ne("ok") if column.name == "box_status" else column


def collect_basketevent(output: str | Path, limit: int | None = None) -> dict:
    """Download labels one game at a time, keep shot rows, delete the JSON; reruns resume."""
    from huggingface_hub import HfApi, snapshot_download
    from nba_api.stats.endpoints import playbyplayv3, shotchartdetail

    output = Path(output)
    games_dir = output / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    log = output / "retrieval.jsonl"
    info = HfApi().dataset_info(REPO)
    files = [s.rfilename for s in info.siblings]
    source = output / "source.json"
    if source.exists() and json.loads(source.read_text())["revision"] != info.sha:
        raise ValueError("BasketEvent changed since collection began; use a new output")
    videos = {f.removesuffix(".mp4") for f in files if f.endswith(".mp4")}
    games = sorted({tuple(f.split("/")[:2]) for f in files if f.count("/") == 2})
    write_json(
        source,
        {"repository": REPO, "revision": info.sha, "season": SEASON, "games": len(games)},
    )
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for split, game in games[:limit]:
        marker = games_dir / f"{game}.json"
        if marker.exists():
            counts["skipped"] += 1
            continue
        record = {"game_id": game, "split": split, "retrieved_at": utc_now()}
        workdir = Path(tempfile.mkdtemp(prefix="basketevent_"))
        try:
            snapshot_download(
                REPO,
                repo_type="dataset",
                revision=info.sha,
                allow_patterns=[f"{split}/{game}/*.json"],
                local_dir=workdir,
            )
            clips = {
                str(p.relative_to(workdir)): json.loads(p.read_text())
                for p in sorted((workdir / split / game).glob("*.json"))
            }
            plays = playbyplayv3.PlayByPlayV3(game_id=game, timeout=30).get_data_frames()[0]
            rows = game_rows(clips, plays)
            raw = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=0,
                game_id_nullable=game,
                season_nullable=SEASON,
                context_measure_simple="FGA",
                season_type_all_star="Regular Season",
                timeout=30,
            ).shot_chart_detail.get_data_frame()
            if raw.empty:
                raise ValueError("NBA returned no shot attempts for this game")
            rows["has_video"] = rows["clip"].str.removesuffix(".json").isin(videos)
            rows["basketevent_split"] = split
            raw.to_csv(games_dir / f"{game}_shots.csv", index=False)
            rows.to_csv(games_dir / f"{game}_clips.csv", index=False)
            record.update(
                {
                    "status": "ok",
                    "clips": len(clips),
                    "shot_events": len(rows),
                    "matched": int(rows.game_event_id.notna().sum()),
                    "release_found": int(rows.box_status.eq("ok").sum()),
                }
            )
            write_json(marker, record)
            counts["done"] += 1
        except Exception as error:
            record.update({"status": "error", "error": str(error)})
            counts["failed"] += 1
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        with log.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        time.sleep(0.6)
    return {**counts, "revision": info.sha, "log": str(log)}


def prepare_basketevent(collection: str | Path, output: str | Path, config: dict) -> dict:
    """Cohort = NBA shot records whose clip has a detected release; box features separately.

    Shots without a release are dropped rather than kept with blank features: a clip holding
    several shots is usually a miss and its putback, so blank features would leak the outcome.
    """
    from nba_api.stats.static import teams

    collection, output = Path(collection), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new cohort directory; do not rewrite frozen game splits")
    shot_files = sorted((collection / "games").glob("*_shots.csv"))
    if not shot_files:
        raise ValueError("No collected games; run collect-basketevent first")
    raw = pd.concat([pd.read_csv(p, dtype={"GAME_ID": str}) for p in shot_files])
    events = pd.concat(
        [
            pd.read_csv(str(p).replace("_shots", "_clips")).assign(game_id=p.name[:10])
            for p in shot_files
        ],
        ignore_index=True,
    )
    abbreviations = {t["id"]: t["abbreviation"] for t in teams.get_teams()}
    shots = normalize_recent(raw.reset_index(drop=True), SEASON, abbreviations)
    matched = events.dropna(subset=["game_event_id"]).copy()
    matched["shot_id"] = (
        "nba:" + matched.game_id + ":" + matched.game_event_id.astype(int).astype(str)
    )
    if matched.shot_id.duplicated().any() or not set(matched.shot_id) <= set(shots.shot_id):
        raise ValueError("Clip events must match logged shot attempts one to one")
    labeled = matched.merge(shots[["shot_id", "y_true"]], on="shot_id")
    matched = matched.loc[matched.box_status.eq("ok")]
    shots = shots.loc[shots.shot_id.isin(matched.shot_id)].assign(cohort=COHORT)
    shots, splits = assign_splits(shots, config["train_fraction"], config["validation_fraction"])
    if config["include_player_history"]:
        shots = add_player_history(shots)
    output.mkdir(parents=True)
    shots.to_parquet(output / "shots.parquet", index=False)
    kept = ["shot_id", "box_status", "basketevent_split", "has_video", "clip", "release_frame"]
    kept += ["shooter_box", "defender_box", *FEATURE_COLUMNS]
    matched[kept].to_parquet(output / "box_features.parquet", index=False)
    write_json(output / "splits.json", splits)
    write_json(output / "config.json", config)
    write_json(
        output / "audit.json",
        {
            "source": json.loads((collection / "source.json").read_text()),
            "prepared_at": utc_now(),
            "games": int(shots.game_id.nunique()),
            "shots": len(shots),
            "make_rate": float(shots.y_true.mean()),
            "date_min": shots.game_date.min(),
            "date_max": shots.game_date.max(),
            "clip_shot_events": len(events),
            # Matching must not succeed more often for makes than for misses.
            "match_rate_by_action": events.groupby("action")
            .game_event_id.apply(lambda s: float(s.notna().mean()))
            .to_dict(),
            "duplicate_clips_dropped": int(events.duplicate_clips.clip(lower=0).sum()),
            # Only "ok" shots enter the cohort; the others are reported, not modeled.
            "status_counts": labeled.box_status.value_counts().to_dict(),
            "make_rate_by_status": labeled.groupby("box_status").y_true.mean().to_dict(),
            "found_rate_by_outcome": labeled.groupby("y_true")
            .box_status.apply(lambda s: float(s.eq("ok").mean()))
            .rename(index=str)
            .to_dict(),
            "with_video": int(matched.has_video.sum()),
            "environment": environment(),
        },
    )
    return {
        "shots": str(output / "shots.parquet"),
        "features": str(output / "box_features.parquet"),
        "rows": len(shots),
    }
