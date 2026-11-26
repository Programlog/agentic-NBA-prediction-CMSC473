"""Independent BasketEvent shooter and shot-event identity review."""

import argparse
import csv
import json
from pathlib import Path

from nba_shots.basketevent import SHOT_ACTIONS
from tools.annotate_release_review import normalize_path, read_csv

FIELDS = [
    "video_path",
    "event_status",
    "shooter_key",
    "player_name",
    "team",
    "jersey_color",
    "action_type",
    "shot_subtype",
    "description",
    "period",
    "clock",
    "other_events",
    "identity_decision",
    "observed_shooter",
    "observed_action",
    "identity_notes",
]

DECISIONS = {"pending", "confirmed", "mismatch", "uncertain"}


def extract_event_identity(clip):
    """Extract metadata without claiming visual verification."""
    shots = []
    other_events = []

    for key, entity in clip.items():
        if not isinstance(entity, dict):
            continue

        event = entity.get("event")
        if not isinstance(event, dict):
            continue

        item = {
            "shooter_key": key,
            "player_name": str(event.get("playerName") or ""),
            "team": str(event.get("teamTricode") or ""),
            "jersey_color": str(entity.get("jersey_color") or ""),
            "action_type": str(event.get("actionType") or ""),
            "shot_subtype": str(event.get("subType") or ""),
            "description": str(event.get("description") or ""),
            "period": str(event.get("period") or ""),
            "clock": str(event.get("clock") or ""),
        }

        if item["action_type"] in SHOT_ACTIONS:
            shots.append(item)
        else:
            other_events.append(
                {
                    "entity": key,
                    "player": item["player_name"],
                    "action": item["action_type"],
                    "clock": item["clock"],
                }
            )

    if not shots:
        status = "no_shot_event"
    elif len(shots) > 1:
        status = "multiple_shot_events"
    else:
        status = "one_shot_event"

    result = {field: "" for field in FIELDS}
    result["event_status"] = status
    result["other_events"] = json.dumps(other_events)
    result["identity_decision"] = "pending"

    if len(shots) == 1:
        result.update(shots[0])

    return result


def build_worksheet(manifest, json_root):
    rows = []
    seen = set()

    for item in manifest:
        video_path = normalize_path(item["video_path"])

        if video_path in seen:
            raise ValueError(f"Duplicate video: {video_path}")
        seen.add(video_path)

        relative_json = Path(video_path).with_suffix(".json")
        json_path = json_root / relative_json

        if not json_path.is_file():
            raise FileNotFoundError(json_path)

        with json_path.open(encoding="utf-8") as stream:
            clip = json.load(stream)

        if not isinstance(clip, dict):
            raise ValueError(f"Expected JSON object: {json_path}")

        row = extract_event_identity(clip)
        row["video_path"] = video_path
        rows.append(row)

    return rows


def validate_identity_rows(rows):
    """Require a human explanation for every non-pending decision."""
    seen = set()

    for row in rows:
        video_path = normalize_path(row["video_path"])

        if video_path in seen:
            raise ValueError(f"Duplicate identity row: {video_path}")
        seen.add(video_path)

        decision = row.get("identity_decision", "")
        if decision not in DECISIONS:
            raise ValueError(f"Invalid identity decision: {decision}")

        if decision == "pending":
            if any(
                (row.get(field) or "").strip()
                for field in (
                    "observed_shooter",
                    "observed_action",
                    "identity_notes",
                )
            ):
                raise ValueError(f"Pending review contains observations: {video_path}")
            continue

        if not (row.get("identity_notes") or "").strip():
            raise ValueError(f"Identity decision needs notes: {video_path}")

        if decision == "confirmed":
            if row.get("event_status") != "one_shot_event":
                raise ValueError(f"Cannot confirm ambiguous shot: {video_path}")

            if not (row.get("observed_shooter") or "").strip():
                raise ValueError(f"Confirmation needs observed shooter: {video_path}")

            if not (row.get("observed_action") or "").strip():
                raise ValueError(f"Confirmation needs observed action: {video_path}")


def write_worksheet(path, rows):
    """Create a new worksheet without overwriting prior evidence."""
    validate_identity_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--json-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate",
        type=Path,
        help="Validate an existing, manually edited identity worksheet",
    )
    args = parser.parse_args()

    if args.validate:
        rows = read_csv(args.validate)
        validate_identity_rows(rows)
        print(f"Validated {len(rows)} identity-review rows")
        return

    if not all((args.manifest, args.json_root, args.output)):
        parser.error("--manifest, --json-root and --output are required")

    rows = build_worksheet(
        read_csv(args.manifest),
        args.json_root,
    )
    write_worksheet(args.output, rows)

    counts = {}
    for row in rows:
        status = row["event_status"]
        counts[status] = counts.get(status, 0) + 1

    print(f"Worksheet: {args.output}")
    print(f"Videos: {len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print("Identity decisions remain pending human review.")


if __name__ == "__main__":
    main()
