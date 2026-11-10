"""Verify the video-shot pool using pinned labels and cached official shot records.

No automatic frame is evaluated here. NBA shot-chart clocks have integer-second
precision; PBP clocks are truncated to that precision, with ambiguous matches
resolved by the official event player name. Labels/outcomes never select rows.
"""

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path("data/release_audit_source")


def normalize_name(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", value.lower())


def main():
    info = json.loads((ROOT / "dataset_info.json").read_text())
    raw_path = Path("data/raw/nba_2025_26/source_shots.csv")
    raw = pd.read_csv(raw_path, dtype={"GAME_ID": str})
    raw["GAME_ID"] = raw.GAME_ID.str.zfill(10)
    raw["seconds"] = raw.MINUTES_REMAINING * 60 + raw.SECONDS_REMAINING
    groups = {key: part for key, part in raw.groupby(["GAME_ID", "PERIOD", "seconds"])}
    rows, unresolved = [], []
    for path in sorted((ROOT / "labels").rglob("*.json")):
        clip = json.loads(path.read_text())
        holders = {}
        for key, entity in clip.items():
            event = entity.get("event") if isinstance(entity, dict) else None
            if isinstance(event, dict) and event.get("actionType") in {"Made Shot", "Missed Shot"}:
                event_key = (event.get("period"), event.get("clock"), event.get("description"))
                holders.setdefault(event_key, []).append(key)
        for (period, clock, description), keys in holders.items():
            event = clip[keys[0]]["event"]
            clock_match = re.fullmatch(r"PT(\d+)M(\d+(?:\.\d+)?)S", clock)
            seconds = int(clock_match[1]) * 60 + int(float(clock_match[2]))
            candidates = groups.get((path.parent.name, period, seconds), raw.iloc[:0])
            if len(candidates) > 1:
                surname = normalize_name(str(event.get("playerName", "")).split(".")[-1])
                candidates = candidates.loc[
                    candidates.PLAYER_NAME.map(normalize_name).str.endswith(surname)
                ]
            if len(candidates) > 1:
                # Resolve event identity only; neither outcomes nor detection quality
                # enter the subsequent sampling rule or the blind review package.
                made = int(event["actionType"] == "Made Shot")
                candidates = candidates.loc[candidates.SHOT_MADE_FLAG.eq(made)]
            if len(candidates) != 1:
                unresolved.append({"clip": str(path), "candidate_count": len(candidates)})
                event_hash = hashlib.sha256(
                    json.dumps([period, clock, description]).encode()
                ).hexdigest()[:16]
                shot_id = f"basketevent:{path.parent.name}:{event_hash}"
            else:
                shot_id = f"nba:{path.parent.name}:{int(candidates.iloc[0].GAME_EVENT_ID)}"
            relative = str(path.relative_to(ROOT / "labels"))
            video_relative = relative.removesuffix(".json") + ".mp4"
            rows.append(
                {
                    "shot_id": shot_id,
                    "official_match_status": "matched" if len(candidates) == 1 else "ambiguous",
                    "game_id": path.parent.name,
                    "clip_path": str((ROOT / "videos" / video_relative).resolve()),
                    "label_path": str(path.resolve()),
                    "shooter_keys": json.dumps(keys),
                    "multiple_events": len(holders) > 1,
                    "source_revision": info["sha"],
                    "source_clip": relative,
                    "video_url": f"https://huggingface.co/datasets/zaywas/BasketEvent/resolve/{info['sha']}/{video_relative}",
                }
            )
    ids = [r["shot_id"] for r in rows]
    audit = {
        "revision": info["sha"],
        "video_files": sum(x["rfilename"].endswith(".mp4") for x in info["siblings"]),
        "video_shot_events": len(rows),
        "official_matched_shots": len(rows) - len(unresolved),
        "source_identified_ambiguous_shots": len(unresolved),
        "unique_shot_ids": len(set(ids)),
        "unresolved": unresolved,
        "official_source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "matching": "game/period/integer-second clock; surname then event outcome break ties",
        "deviation": "Live PBP unavailable (403/timeout); cached official ShotChartDetail used.",
    }
    (ROOT / "pool_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Population matching incomplete: {len(unresolved)} unresolved; "
            f"{len(ids) - len(set(ids))} duplicate IDs"
        )
    with (ROOT / "pool.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in audit.items() if k != "unresolved"}))


if __name__ == "__main__":
    main()
