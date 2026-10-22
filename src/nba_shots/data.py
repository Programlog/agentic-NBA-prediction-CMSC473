"""Source acquisition, explicit feature selection and chronological game splitting."""

from __future__ import annotations

import hashlib
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .common import sha256, utc_now, write_json

KAGGLE_SOURCE = "https://www.kaggle.com/datasets/dansbecker/nba-shot-logs"
KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/dansbecker/nba-shot-logs"
NUMERIC_MAPPING = {
    "SHOT_DIST": "shot_distance",
    "CLOSE_DEF_DIST": "defender_distance",
    "SHOT_CLOCK": "shot_clock",
    "PERIOD": "period",
    "DRIBBLES": "dribbles",
    "TOUCH_TIME": "touch_time",
    "PTS_TYPE": "shot_value",
}


def download_kaggle(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "shots.zip"
        with requests.get(KAGGLE_DOWNLOAD, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with archive.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
            resolved_url = response.url
        if not zipfile.is_zipfile(archive):
            raise ValueError("Kaggle returned no ZIP. Download shot_logs.csv and use --input.")
        with zipfile.ZipFile(archive) as bundle:
            members = [n for n in bundle.namelist() if Path(n).name == "shot_logs.csv"]
            if len(members) != 1:
                raise ValueError("Expected exactly one shot_logs.csv in the Kaggle archive")
            contents = bundle.read(members[0])
            # Reject HTML/auth responses before persisting the dataset.
            if b"SHOT_RESULT" not in contents.splitlines()[0]:
                raise ValueError("Downloaded CSV does not have the expected shot-log header")
            destination.write_bytes(contents)
        write_json(
            destination.with_suffix(".source.json"),
            {
                "source": KAGGLE_SOURCE,
                "download_url": KAGGLE_DOWNLOAD,
                "resolved_url": resolved_url,
                "retrieved_at": utc_now(),
                "archive_sha256": sha256(archive),
                "csv_sha256": sha256(destination),
            },
        )
    return destination


def normalize_game_id(value) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d{1,10}", text):
        raise ValueError(f"Invalid NBA game ID: {value!r}")
    return text.zfill(10)


def parse_clock(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"(\d{1,2}):([0-5]\d)(?:\.\d+)?", str(value).strip())
    if not match:
        return np.nan
    return int(match[1]) * 60 + float(str(value).strip().split(":")[1])


def normalize_kaggle(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "GAME_ID",
        "MATCHUP",
        "LOCATION",
        "GAME_CLOCK",
        "SHOT_RESULT",
        "player_id",
        "SHOT_NUMBER",
        *NUMERIC_MAPPING,
    }
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing Kaggle columns: {sorted(missing)}")
    labels = raw["SHOT_RESULT"].astype(str).str.lower().str.strip()
    if not labels.isin(["made", "missed"]).all():
        raise ValueError("SHOT_RESULT must contain only made/missed; labels are never imputed")
    dates = pd.to_datetime(
        raw["MATCHUP"].str.extract(r"^([A-Z]{3}\s+\d{1,2},\s*\d{4})", expand=False),
        format="%b %d, %Y",
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError("Every shot needs a valid game date for chronological splitting")
    frame = pd.DataFrame(
        {
            "game_id": raw["GAME_ID"].map(normalize_game_id),
            "game_date": dates.dt.strftime("%Y-%m-%d"),
            "player_id": raw["player_id"].astype(str),
            "cohort": "kaggle_2014_15",
            "y_true": labels.eq("made").astype(int),
        }
    )
    if raw[["player_id", "SHOT_NUMBER"]].isna().any().any():
        raise ValueError("Shot identity fields cannot be missing")
    frame["shot_id"] = (
        "kaggle:"
        + frame["game_id"]
        + ":"
        + frame["player_id"]
        + ":"
        + raw["SHOT_NUMBER"].astype(str)
    )
    if frame["shot_id"].duplicated().any():
        raise ValueError("Duplicate game/player/shot-number keys; resolve source duplicates first")
    invalid = {}
    for source, name in NUMERIC_MAPPING.items():
        values = pd.to_numeric(raw[source], errors="coerce").astype(float)
        bad = ~np.isfinite(values) | values.lt(0)
        if name == "shot_clock":
            bad |= values.gt(24)
        elif name == "period":
            bad |= values.lt(1) | values.mod(1).ne(0)
        elif name == "shot_value":
            bad |= ~values.isin([2, 3])
        invalid[name] = int((bad & raw[source].notna()).sum())
        frame[name] = values.mask(bad)
    clock = raw["GAME_CLOCK"].map(parse_clock)
    max_clock = np.where(frame["period"].le(4), 720, 300)
    frame["game_clock_seconds"] = clock.mask(clock.gt(max_clock))
    location = raw["LOCATION"].astype("string").str.upper()
    frame["location"] = location.where(location.isin(["H", "A"]), np.nan).astype(object)
    frame = frame.sort_values(["game_date", "game_id", "shot_id"]).reset_index(drop=True)
    audit = {
        "raw_rows": len(raw),
        "prepared_rows": len(frame),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_shot_ids": 0,
        "games": int(frame["game_id"].nunique()),
        "dates": int(frame["game_date"].nunique()),
        "date_min": frame["game_date"].min(),
        "date_max": frame["game_date"].max(),
        "make_rate": float(frame["y_true"].mean()),
        "raw_missing_values": {k: int(v) for k, v in raw.isna().sum().items()},
        "feature_missing_values": {k: int(v) for k, v in frame.isna().sum().items() if v},
        "invalid_numeric_values_replaced_with_missing": invalid,
        "target": "SHOT_RESULT: made=1, missed=0",
        "excluded_outcome_columns": ["FGM", "PTS", "W", "FINAL_MARGIN"],
    }
    return frame, audit


def validate_shots(frame: pd.DataFrame) -> None:
    required = {"shot_id", "game_id", "game_date", "cohort", "y_true"}
    if missing := required - set(frame):
        raise ValueError(f"Missing shot metadata: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Shot metadata must be present and nonempty")
    if frame["shot_id"].duplicated().any():
        raise ValueError("shot_id must be unique")
    if not frame["y_true"].isin([0, 1]).all():
        raise ValueError("Targets must be binary")
    if frame["cohort"].nunique() != 1:
        raise ValueError("Prepare/evaluate cohorts separately; do not pool different datasets")
    pd.to_datetime(frame["game_date"], format="%Y-%m-%d", errors="raise")
    if frame.groupby("game_id")["game_date"].nunique().gt(1).any():
        raise ValueError("A game must have exactly one date")


def assign_splits(frame: pd.DataFrame, train_fraction=0.7, validation_fraction=0.15):
    validate_shots(frame)
    if not (
        0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Invalid split fractions")
    dates = sorted(frame["game_date"].unique())
    if len(dates) < 3:
        raise ValueError("At least three distinct game dates are required")
    counts = frame.groupby("game_date")["game_id"].nunique().reindex(dates).cumsum()
    # Choose whole-date boundaries nearest the requested proportions of games.
    a = min(
        range(len(dates) - 2), key=lambda i: abs(counts.iloc[i] / counts.iloc[-1] - train_fraction)
    )
    b = min(
        range(a + 1, len(dates) - 1),
        key=lambda i: abs(counts.iloc[i] / counts.iloc[-1] - train_fraction - validation_fraction),
    )
    membership = {
        d: "train" if i <= a else "validation" if i <= b else "test" for i, d in enumerate(dates)
    }
    result = frame.copy()
    result["split"] = result["game_date"].map(membership)
    manifest = {
        "method": "chronological whole dates, nearest requested fraction of games",
        "requested_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1 - train_fraction - validation_fraction,
        },
        "games": {
            split: sorted(result.loc[result["split"].eq(split), "game_id"].unique())
            for split in ["train", "validation", "test"]
        },
        "summary": {},
    }
    for split, part in result.groupby("split"):
        manifest["summary"][split] = {
            "shots": len(part),
            "games": int(part["game_id"].nunique()),
            "date_min": part["game_date"].min(),
            "date_max": part["game_date"].max(),
        }
    return result, manifest


def add_player_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Use only completed earlier-date games, never another shot from the same date."""
    if "player_id" not in frame:
        raise ValueError("player_id is required for historical features")
    daily = (
        frame.groupby(["player_id", "game_date"], as_index=False)
        .agg(
            attempts=("y_true", "size"),
            makes=("y_true", "sum"),
        )
        .sort_values(["player_id", "game_date"])
    )
    groups = daily.groupby("player_id", sort=False)
    daily["player_prior_attempts"] = groups["attempts"].cumsum() - daily["attempts"]
    prior_makes = groups["makes"].cumsum() - daily["makes"]
    # A fixed Beta(1,1) prior avoids learning a prior from future games.
    daily["player_prior_make_rate"] = (prior_makes + 1) / (daily["player_prior_attempts"] + 2)
    return frame.merge(
        daily[["player_id", "game_date", "player_prior_attempts", "player_prior_make_rate"]],
        on=["player_id", "game_date"],
        how="left",
        validate="many_to_one",
    )


def prepare_csv(input_path: str | Path, output_dir: str | Path, config: dict) -> Path:
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "shots.parquet").exists():
        raise ValueError(f"Prepared data already exists in {output_dir}; choose a new directory")
    raw = pd.read_csv(input_path, dtype={"GAME_ID": str, "player_id": str, "SHOT_NUMBER": str})
    shots, audit = normalize_kaggle(raw)
    shots, manifest = assign_splits(shots, config["train_fraction"], config["validation_fraction"])
    if config["include_player_history"]:
        shots = add_player_history(shots)
    audit.update(
        {
            "input": str(input_path.resolve()),
            "csv_sha256": sha256(input_path),
            "source": KAGGLE_SOURCE,
            "prepared_at": utc_now(),
        }
    )
    manifest["source_sha256"] = sha256(input_path)
    manifest["shot_ids_sha256"] = hashlib.sha256(
        "\n".join(sorted(shots["shot_id"])).encode()
    ).hexdigest()
    shots.to_parquet(output_dir / "shots.parquet", index=False)
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "splits.json", manifest)
    write_json(output_dir / "config.json", config)
    return output_dir / "shots.parquet"
