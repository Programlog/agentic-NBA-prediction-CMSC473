
import re
from pathlib import Path

import pandas as pd

BASE = Path("data/basketevent")
CANDIDATES = BASE / "shot_candidates.csv"
NBA_SHOTS = Path("data/recent/shots.parquet")
OUTPUT = BASE / "batch_01_manifest.csv"
REPORT = BASE / "matching_report.csv"


def parse_clock(clock):
    """Convert PT08M04.00S to 484 seconds."""
    match = re.fullmatch(
        r"PT(\d+)M(\d+(?:\.\d+)?)S",
        str(clock),
    )
    if not match:
        raise ValueError(f"Unexpected clock: {clock}")
    minutes = int(match.group(1))
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def parse_distance(description):
    """Extract approximate shot distance from NBA play text."""
    match = re.search(r"(\d+)'", str(description))
    return int(match.group(1)) if match else None


def match_shot(candidate, nba):
    game_id = str(candidate["game_id"]).zfill(10)
    period = int(candidate["period"])
    clock = parse_clock(candidate["clock"])
    made = int(
        str(candidate["action_type"]).strip().lower() == "made shot"
    )

    subset = nba[
        (nba["game_id"] == game_id)
        & (nba["period"] == period)
        & (nba["y_true"] == made)
    ].copy()

    # First require exact game-clock matching.
    subset = subset[
        (subset["game_clock_seconds"] - clock).abs() < 0.01
    ]

    # An exact timestamp must resolve to one candidate.
    # Never silently choose between multiple NBA events.
    if len(subset) != 1:
        return None, f"Expected 1 exact match; found {len(subset)}"

    shot = subset.iloc[0]
    description = str(candidate["description"])
    distance = parse_distance(description)

    # Check that the play description is consistent with
    # the shot record. A small distance difference is allowed.
    if distance is not None:
        if abs(float(shot["shot_distance"]) - distance) > 2:
            return None, (
                f"Distance mismatch: description={distance}, "
                f"NBA={shot['shot_distance']}"
            )

    # Independently verify three-point descriptions.
    if "3PT" in description.upper() and int(shot["shot_value"]) != 3:
        return None, "Shot-value mismatch: expected three-pointer"

    return shot, "matched"


def main():
    if OUTPUT.exists():
        raise SystemExit(
            f"{OUTPUT} already exists. Preserve the old manifest."
        )

    candidates = pd.read_csv(
        CANDIDATES,
        dtype={"game_id": str},
    )
    nba = pd.read_parquet(NBA_SHOTS)

    nba["game_id"] = nba["game_id"].astype(str).str.zfill(10)

    manifest_rows = []
    report_rows = []
    used_shot_ids = set()

    for _, candidate in candidates.iterrows():
        try:
            shot, status = match_shot(candidate, nba)
        except (ValueError, TypeError, KeyError) as exc:
            shot, status = None, f"Error: {exc}"

        report = {
            "player": candidate["player"],
            "game_id": candidate["game_id"],
            "period": candidate["period"],
            "clock": candidate["clock"],
            "description": candidate["description"],
            "action_type": candidate["action_type"],
            "video_path": candidate["video_path"],
            "shot_id": "",
            "nba_player_id": "",
            "status": status,
        }

        if shot is not None:
            shot_id = str(shot["shot_id"])

            if shot_id in used_shot_ids:
                report["status"] = "Duplicate shot ID"
            else:
                clip = BASE / str(candidate["video_path"])

                if not clip.is_file():
                    report["status"] = "Video file missing"
                else:
                    used_shot_ids.add(shot_id)
                    report["shot_id"] = shot_id
                    report["nba_player_id"] = str(
                        shot["player_id"]
                    )

                    manifest_rows.append({
                        "shot_id": shot_id,
                        "game_id": str(shot["game_id"]),
                        "game_event_id": str(shot["game_event_id"]),
                        "clip_path": str(candidate["video_path"]),
                        "release_frame": "",
                        "shooter_box": "",
                        "defender_box": "",
                        "shooting_side": "",
                    })

        report_rows.append(report)
        print(
            f"{candidate['player']} | "
            f"{candidate['clock']} | "
            f"{report['status']} | "
            f"{report['shot_id']}"
        )

    pd.DataFrame(report_rows).to_csv(REPORT, index=False)

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(OUTPUT, index=False)

    print("\n========== MATCHING SUMMARY ==========")
    print("Total candidates:", len(candidates))
    print("Matched shots:", len(manifest_rows))
    print("Unmatched:", len(candidates) - len(manifest_rows))
    print("Report:", REPORT)

    if manifest_rows:
        print("Annotation manifest:", OUTPUT)


if __name__ == "__main__":
    main()
