"""Matched shot/video cohorts and release-frame-only 2D pose measurements."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .common import CV_FEATURES, environment, sha256, utc_now, write_json
from .data import add_player_history, assign_splits, normalize_game_id, validate_shots


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"shot_id": str, "game_id": str, "game_event_id": str})
    if "shot_id" not in frame or frame.empty or frame.shot_id.isna().any():
        raise ValueError("Manifest must contain nonempty shot_id values")
    if frame.shot_id.duplicated().any():
        raise ValueError("Manifest shot_id values must be unique")
    return frame


def collect_recent(season: str, output: str | Path) -> dict:
    from nba_api.stats.endpoints import shotchartdetail
    from nba_api.stats.static import teams

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Collection directory must be new; cached source records are immutable")
    response = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,
        season_nullable=season,
        context_measure_simple="FGA",
        season_type_all_star="Regular Season",
        timeout=30,
    )
    raw = response.shot_chart_detail.get_data_frame()
    if raw.empty:
        raise ValueError("NBA returned no shot attempts; do not proceed with an empty cohort")
    frame = normalize_recent(raw, season, {t["id"]: t["abbreviation"] for t in teams.get_teams()})
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "source_shots.csv", index=False)
    frame.to_parquet(output / "shots.parquet", index=False)
    write_json(
        output / "source.json",
        {
            "source": "NBA.com ShotChartDetail via nba_api",
            "season": season,
            "retrieved_at": utc_now(),
            "source_sha256": sha256(output / "source_shots.csv"),
            "rows": len(frame),
            "games": int(frame.game_id.nunique()),
            "date_min": frame.game_date.min(),
            "date_max": frame.game_date.max(),
            "possible_row_cap": len(raw) >= 100000,
            "coverage_note": "A source snapshot, not a guarantee of full-season coverage",
            "unavailable_features": ["shot_clock", "defender_distance", "dribbles", "touch_time"],
            "environment": environment(),
        },
    )
    return {"shots": str(output / "shots.parquet"), "rows": len(frame)}


def normalize_recent(raw: pd.DataFrame, season: str, abbreviations: dict) -> pd.DataFrame:
    required = {
        "GAME_ID",
        "GAME_EVENT_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "TEAM_ID",
        "HTM",
        "SHOT_MADE_FLAG",
        "SHOT_DISTANCE",
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "SHOT_TYPE",
    }
    if missing := required - set(raw):
        raise ValueError(f"NBA response missing columns: {sorted(missing)}")
    frame = pd.DataFrame(
        {
            "game_id": raw.GAME_ID.map(normalize_game_id),
            "game_event_id": raw.GAME_EVENT_ID.astype(str),
            "game_date": pd.to_datetime(raw.GAME_DATE, format="%Y%m%d").dt.strftime("%Y-%m-%d"),
            "player_id": raw.PLAYER_ID.astype(str),
            "cohort": f"nba_{season}",
            "y_true": pd.to_numeric(raw.SHOT_MADE_FLAG),
            "shot_distance": pd.to_numeric(raw.SHOT_DISTANCE),
            "period": pd.to_numeric(raw.PERIOD),
            "game_clock_seconds": pd.to_numeric(raw.MINUTES_REMAINING) * 60
            + pd.to_numeric(raw.SECONDS_REMAINING),
            "shot_value": raw.SHOT_TYPE.map({"2PT Field Goal": 2, "3PT Field Goal": 3}),
            "location": np.where(raw.TEAM_ID.map(abbreviations).eq(raw.HTM), "H", "A"),
        }
    )
    frame["shot_id"] = "nba:" + frame.game_id + ":" + frame.game_event_id
    for name in ["defender_distance", "shot_clock", "dribbles", "touch_time"]:
        frame[name] = np.nan
    validate_shots(frame)
    return frame.sort_values(["game_date", "game_id", "shot_id"]).reset_index(drop=True)


def make_manifest(shots_path, output, games=30, shots_per_game=35, seed=42) -> dict:
    shots, output = pd.read_parquet(shots_path), Path(output)
    validate_shots(shots)
    if "game_event_id" not in shots:
        raise ValueError("Use recent shot records with game_event_id for video matching")
    if games < 3 or shots_per_game < 1:
        raise ValueError("Select at least three games and one shot per game")
    if output.exists():
        raise ValueError("Manifest already exists; do not overwrite annotations")

    def rank(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    selected_games = sorted(shots.game_id.unique(), key=rank)[:games]
    rows = []
    for game in selected_games:
        part = shots.loc[shots.game_id.eq(game)].copy()
        ids = sorted(part.shot_id, key=rank)[:shots_per_game]
        rows.append(part.loc[part.shot_id.isin(ids)])
    selected = pd.concat(rows).sort_values(["game_date", "game_id", "shot_id"])
    if selected.game_date.nunique() < 3:
        raise ValueError("Sample needs at least three dates for chronological evaluation")
    manifest = selected[["shot_id", "game_id", "game_event_id"]].copy()
    for column in ["clip_path", "release_frame", "shooter_box", "defender_box", "shooting_side"]:
        manifest[column] = ""
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    write_json(
        output.with_suffix(".selection.json"),
        {
            "shots_sha256": sha256(shots_path),
            "manifest_sha256": sha256(output),
            "seed": seed,
            "games": len(selected_games),
            "shots": len(manifest),
            "selection": "Hash-seeded game sampling, then hash-seeded shots; no outcome selection",
        },
    )
    return {"manifest": str(output), "games": len(selected_games), "shots": len(manifest)}


def prepare_cohort(shots_path, manifest_path, output, config) -> dict:
    shots, manifest = pd.read_parquet(shots_path), load_manifest(manifest_path)
    validate_shots(shots)
    if not set(manifest.shot_id) <= set(shots.shot_id):
        raise ValueError("Every preregistered shot ID must exist in the source records")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new cohort directory; do not rewrite frozen game splits")
    if config["include_player_history"]:
        shots = add_player_history(shots)
    # Register attempted shots, including unavailable videos and failed pose extraction.
    cohort = shots.loc[shots.shot_id.isin(manifest.shot_id)].copy()
    if "game_id" in manifest:
        joined = manifest.merge(cohort[["shot_id", "game_id"]], on="shot_id", suffixes=("", "_src"))
        if not joined.game_id.map(normalize_game_id).eq(joined.game_id_src).all():
            raise ValueError("Manifest game IDs do not match the shot source")
    cohort, split_manifest = assign_splits(
        cohort,
        config["train_fraction"],
        config["validation_fraction"],
    )
    output.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output / "shots.parquet", index=False)
    write_json(output / "splits.json", split_manifest)
    write_json(output / "config.json", config)
    write_json(
        output / "audit.json",
        {
            "source_sha256": sha256(shots_path),
            "manifest_sha256": sha256(manifest_path),
            "registered_shots": len(cohort),
            "games": int(cohort.game_id.nunique()),
            "note": "No outcome- or pose-quality-based filtering; keep failed attempts in CV table",
        },
    )
    return {"shots": str(output / "shots.parquet"), "registered_shots": len(cohort)}


def _video_url(value):
    if isinstance(value, dict):
        for name in ["lurl", "murl", "surl"]:
            if isinstance(value.get(name), str) and value[name].startswith("https://"):
                return value[name]
        for child in value.values():
            if found := _video_url(child):
                return found
    elif isinstance(value, list):
        for child in value:
            if found := _video_url(child):
                return found
    return None


def fetch_videos(manifest_path, output, limit=20) -> dict:
    from nba_api.stats.endpoints import videoeventsasset

    if limit < 1:
        raise ValueError("Video limit must be positive")
    manifest, output = load_manifest(manifest_path), Path(output)
    if not {"game_id", "game_event_id"} <= set(manifest):
        raise ValueError("Video retrieval needs game_id and game_event_id in the manifest")
    output.mkdir(parents=True, exist_ok=True)
    log = output / "retrieval.jsonl"
    counts = {"downloaded_or_cached": 0, "failed": 0}
    for row in manifest.head(limit).to_dict("records"):
        record = {"shot_id": row["shot_id"], "retrieved_at": utc_now()}
        clip = output / (row["shot_id"].replace(":", "_") + ".mp4")
        partial = clip.with_suffix(".part")
        try:
            if not clip.exists():
                response = videoeventsasset.VideoEventsAsset(
                    game_id=normalize_game_id(row["game_id"]),
                    game_event_id=row["game_event_id"],
                    timeout=20,
                ).get_dict()
                url = _video_url(response)
                if not url:
                    raise ValueError("No video URL returned")
                record["url"] = url
                with requests.get(url, stream=True, timeout=(15, 60)) as download:
                    download.raise_for_status()
                    size = 0
                    with partial.open("wb") as stream:
                        for block in download.iter_content(1024 * 1024):
                            size += len(block)
                            if size > 100 * 1024 * 1024:
                                raise ValueError("Clip exceeded 100 MiB limit")
                            stream.write(block)
                partial.rename(clip)
            record.update(
                {
                    "status": "downloaded_needs_manual_review",
                    "clip_path": str(clip),
                    "sha256": sha256(clip),
                }
            )
            counts["downloaded_or_cached"] += 1
        except Exception as error:
            partial.unlink(missing_ok=True)
            record.update({"status": "error", "error": str(error)})
            counts["failed"] += 1
        with log.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        time.sleep(0.6)
    return {
        **counts,
        "log": str(log),
        "note": "Review every downloaded clip: a placeholder can be a valid MP4",
    }


def read_release_frame(path: Path, release_frame: int, cv2):
    """Read only the annotated frame; no trajectory, follow-through or outcome frames."""
    if release_frame < 0:
        raise ValueError("release_frame must be a nonnegative zero-based frame index")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Video cannot be opened")
        capture.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
        ok, image = capture.read()
        actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if not ok or actual != release_frame:
            raise ValueError(f"Requested frame {release_frame}, decoded frame {actual}")
        return image
    finally:
        capture.release()


def match_box(boxes: np.ndarray, annotation) -> int:
    box = np.asarray(
        json.loads(annotation) if isinstance(annotation, str) else annotation, dtype=float
    )
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Person box must be [x1,y1,x2,y2] with positive area")
    boxes = np.asarray(boxes, dtype=float)
    if not len(boxes):
        raise ValueError("No detected people")
    intersection = np.maximum(
        0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2])
    ).prod(axis=1)
    union = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) + np.prod(box[2:] - box[:2]) - intersection
    iou = intersection / np.maximum(union, 1e-9)
    index = int(iou.argmax())
    if iou[index] < 0.2:
        raise ValueError("Annotated person cannot be matched to a detected pose")
    return index


def pose_features(
    shooter: np.ndarray, defender: np.ndarray | None = None, side="right", confidence=0.5
) -> dict:
    """Keypoints are 17 x [x,y,confidence]; distances are torso-normalized image units."""
    if side not in {"left", "right"}:
        raise ValueError("shooting_side must be left/right")
    shooter = np.asarray(shooter, dtype=float)
    if shooter.shape != (17, 3):
        raise ValueError("Expected 17 COCO keypoints with confidence")

    def point(pose, i):
        return pose[i, :2] if pose[i, 2] >= confidence and np.isfinite(pose[i]).all() else None

    def center(pose, a, b):
        points = [point(pose, a), point(pose, b)]
        return np.mean(points, axis=0) if all(p is not None for p in points) else None

    result = {name: np.nan for name in sorted(CV_FEATURES)}
    shoulders, hips = center(shooter, 5, 6), center(shooter, 11, 12)
    if shoulders is None or hips is None:
        return result
    torso = np.linalg.norm(shoulders - hips)
    if torso < 1e-6:
        return result
    s, e, w = (5, 7, 9) if side == "left" else (6, 8, 10)
    shoulder, elbow, wrist = [point(shooter, i) for i in [s, e, w]]
    if wrist is not None:
        result["wrist_height_torso"] = float((shoulders[1] - wrist[1]) / torso)
    if all(p is not None for p in [shoulder, elbow, wrist]):
        a, b = shoulder - elbow, wrist - elbow
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator > 1e-9:
            result["elbow_angle_deg"] = float(
                np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1)))
            )
    if defender is not None:
        defender = np.asarray(defender, dtype=float)
        if defender.shape != (17, 3):
            raise ValueError("Expected 17 defender keypoints with confidence")
        defender_hips = center(defender, 11, 12)
        if defender_hips is not None:
            result["defender_distance_torso"] = float(np.linalg.norm(defender_hips - hips) / torso)
        wrists = [point(defender, i) for i in [9, 10]]
        wrists = [p for p in wrists if p is not None]
        if wrists:
            result["defender_wrist_height_torso"] = float(
                (shoulders[1] - min(p[1] for p in wrists)) / torso
            )
    return result


def extract_cv(manifest_path, output, weights="yolo11n-pose.pt", device="cpu") -> dict:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("artifacts/yolo_state").resolve()))
    import cv2
    from ultralytics import YOLO

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = load_manifest(manifest_path)
    required = {"clip_path", "release_frame", "shooter_box", "shooting_side"}
    if missing := required - set(manifest):
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new extraction directory to preserve CV provenance")
    model = YOLO(weights)
    output.mkdir(parents=True, exist_ok=True)
    audit_dir = output / "audit_frames"
    audit_dir.mkdir()
    records = []
    for row in manifest.to_dict("records"):
        record = {
            "shot_id": row["shot_id"],
            **{k: np.nan for k in sorted(CV_FEATURES)},
            "cv_status": "error",
            "cv_error": None,
            "latest_frame_used": None,
        }
        try:
            index = float(row["release_frame"])
            if not np.isfinite(index) or not index.is_integer():
                raise ValueError("release_frame must be an integer")
            path = Path(str(row["clip_path"]))
            if not path.is_absolute():
                path = manifest_path.parent / path
            if not path.is_file():
                raise ValueError("Clip is missing; retain this shot with missing CV features")
            image = read_release_frame(path, int(index), cv2)
            prediction = model.predict(image, device=device, verbose=False)[0]
            if prediction.keypoints is None:
                raise ValueError("No pose keypoints returned")
            boxes = prediction.boxes.xyxy.cpu().numpy()
            points = prediction.keypoints.data.cpu().numpy()
            shooter_index = match_box(boxes, row["shooter_box"])
            defender = None
            if pd.notna(row.get("defender_box")):
                defender_index = match_box(boxes, row["defender_box"])
                if defender_index == shooter_index:
                    raise ValueError("Shooter and defender cannot be the same detection")
                defender = points[defender_index]
            features = pose_features(points[shooter_index], defender, row["shooting_side"])
            record.update(features)
            record.update(
                {
                    "cv_status": "ok"
                    if all(np.isfinite(v) for v in features.values())
                    else "partial",
                    "latest_frame_used": int(index),
                    "release_frame": int(index),
                    "clip_sha256": sha256(path),
                }
            )
            if len(records) < 50:
                overlay = prediction.plot()
                cv2.imwrite(str(audit_dir / (row["shot_id"].replace(":", "_") + ".jpg")), overlay)
        except Exception as error:
            record["cv_error"] = str(error)
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_parquet(output / "cv_features.parquet", index=False)
    counts = {str(k): int(v) for k, v in frame.cv_status.value_counts().items()}
    write_json(
        output / "extraction.json",
        {
            "created_at": utc_now(),
            "manifest_sha256": sha256(manifest_path),
            "weights": weights,
            "weights_sha256": sha256(model.ckpt_path),
            "device": device,
            "counts": counts,
            "environment": environment(),
            "definitions": "Release-frame 2D pose proxies; no physical height or ball trajectory",
        },
    )
    return {"features": str(output / "cv_features.parquet"), "counts": counts}
