"""Audit event metadata coverage in one BasketEvent game without downloading videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_basketevent_multigame import extract_events, indexed_paths, stable_order
from huggingface_hub import HfApi, hf_hub_download


def count_event_metadata(data: dict) -> tuple[int, int, Counter]:
    """Count event fields and non-null event dictionaries in one JSON file."""
    event_field_entries = 0
    non_null_event_entries = 0
    action_counts = Counter()

    for annotation in data.values():
        if not isinstance(annotation, dict):
            continue

        if "event" in annotation:
            event_field_entries += 1

        event = annotation.get("event")
        if isinstance(event, dict):
            non_null_event_entries += 1
            action_counts[str(event.get("actionType", "<missing>")).strip()] += 1

    return event_field_entries, non_null_event_entries, action_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repository = "zaywas/BasketEvent"
    revision = "c8629460c68e6ba65375d2c4f6d8638650b23117"

    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")

    api = HfApi()
    files = api.list_repo_files(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
    )

    groups = indexed_paths(files)
    if args.game not in groups:
        raise ValueError(f"Game not found: {args.game}")

    ordered = sorted(
        groups[args.game],
        key=lambda path: stable_order(args.seed, path),
    )

    records = []
    action_counts = Counter()

    for index, name in enumerate(ordered, 1):
        record = {
            "game_id": args.game,
            "json_path": name,
            "status": "error",
            "event_field_entries": 0,
            "non_null_event_entries": 0,
            "recognized_shot_events": 0,
            "error": "",
        }

        try:
            local = hf_hub_download(
                repo_id=repository,
                repo_type="dataset",
                filename=name,
                revision=revision,
                local_dir="data/basketevent",
            )

            data = json.loads(Path(local).read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                raise ValueError("JSON root is not an object")

            event_fields, non_null_events, file_actions = count_event_metadata(data)
            record["event_field_entries"] = event_fields
            record["non_null_event_entries"] = non_null_events
            action_counts.update(file_actions)

            record["recognized_shot_events"] = len(extract_events(data, name))
            record["status"] = "ok"

        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"

        records.append(record)

        if index % 25 == 0 or index == len(ordered):
            print(f"Processed {index}/{len(ordered)} JSON files", flush=True)

    args.output.mkdir(parents=True)

    with (args.output / "annotation_coverage.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "repository": repository,
        "dataset_revision": revision,
        "game_id": args.game,
        "seed": args.seed,
        "total_paired_json_files": len(ordered),
        "json_failed": sum(r["status"] != "ok" for r in records),
        "files_with_event_metadata": sum(r["event_field_entries"] > 0 for r in records),
        "files_with_non_null_event_metadata": sum(r["non_null_event_entries"] > 0 for r in records),
        "files_with_recognized_shots": sum(r["recognized_shot_events"] > 0 for r in records),
        "recognized_shot_events": sum(r["recognized_shot_events"] for r in records),
        "action_type_counts": dict(action_counts),
        "audit_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
