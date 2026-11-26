"""Reproducible, revision-pinned BasketEvent trajectory audit."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from nba_shots.release_diagnostics import diagnose_clip

REPO = "zaywas/BasketEvent"
REVISION = "d28cf284f2c4cc193bf9c88ccf85524193993984"
SEED = 473

ELIGIBLE_STATUSES = {
    "multiple_candidate_episodes",
    "single_candidate_episode",
    "release_not_found",
}


def select_sample(paths, splits, per_split, seed):
    """Select deterministic samples independently from each split."""
    if per_split <= 0:
        raise ValueError("per_split must be positive")

    groups = defaultdict(list)

    for path in paths:
        if not path.endswith(".json"):
            continue
        split = path.split("/")[0]
        if split in splits:
            groups[split].append(path)

    selected = []

    for split in splits:
        candidates = sorted(set(groups[split]))

        if len(candidates) < per_split:
            raise ValueError(
                f"{split}: requested {per_split} files, only {len(candidates)} available"
            )

        rng = random.Random(seed)
        selected.extend(rng.sample(candidates, per_split))

    return selected


def audit_sample(selected, root, revision, download):
    """Audit only manifest-selected files, never an entire local directory."""
    records = []

    for index, filename in enumerate(selected, start=1):
        relative = Path(filename)

        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".json":
            raise ValueError(f"Unsafe JSON path: {filename}")

        destination = root / relative

        if not destination.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {destination}; rerun with --download")

            destination.parent.mkdir(parents=True, exist_ok=True)

            hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                revision=revision,
                filename=filename,
                local_dir=root,
            )

        raw_bytes = destination.read_bytes()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()
        clip = json.loads(raw_bytes)

        result = diagnose_clip(clip)

        records.append(
            {
                "clip_path": relative.as_posix(),
                "sha256": file_hash,
                "split": relative.parts[0],
                "status": result["status"],
                "eligible": result["status"] in ELIGIBLE_STATUSES,
                "raw_geometry_candidate_count": result.get("raw_geometry_candidate_count", 0),
                "candidate_count": result.get("candidate_count", 0),
                "candidates_removed_by_steady": result.get("candidates_removed_by_steady", 0),
            }
        )

        if index % 50 == 0:
            print(f"Processed {index}/{len(selected)}")

    return records


def summarize(records, splits):
    summary = []

    for split in splits:
        subset = [record for record in records if record["split"] == split]

        statuses = Counter(record["status"] for record in subset)

        eligible = [record for record in subset if record["eligible"]]

        affected = [
            record for record in eligible if int(record["candidates_removed_by_steady"]) > 0
        ]

        lost_all = [
            record
            for record in eligible
            if int(record["raw_geometry_candidate_count"]) > 0
            and int(record["candidate_count"]) == 0
        ]

        summary.append(
            {
                "split": split,
                "sampled_files": len(subset),
                "eligible_shot_clips": len(eligible),
                "affected_clips": len(affected),
                "candidate_frames_removed": sum(
                    int(record["candidates_removed_by_steady"]) for record in eligible
                ),
                "complete_candidate_loss_clips": len(lost_all),
                "no_shot_event": statuses["no_shot_event"],
                "ambiguous_shot_event": statuses["ambiguous_shot_event"],
                "release_not_found": statuses["release_not_found"],
                "single_candidate_episode": statuses["single_candidate_episode"],
                "multiple_candidate_episodes": statuses["multiple_candidate_episodes"],
            }
        )

    return summary


def write_csv(path, fields, rows):
    """Refuse to overwrite prior research outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/basketevent"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/expanded_trajectory_audit"),
    )
    parser.add_argument("--per-split", type=int, default=200)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if len(args.splits) != len(set(args.splits)):
        parser.error("Duplicate splits are not allowed")

    if not set(args.splits).issubset({"train", "valid", "test"}):
        parser.error("Invalid split")

    paths = HfApi().list_repo_files(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
    )

    selected = select_sample(
        paths,
        args.splits,
        args.per_split,
        SEED,
    )

    records = audit_sample(selected, args.root, REVISION, args.download)
    summary = summarize(records, args.splits)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # Check output paths before writing any results.
    outputs = {
        "manifest": output / "manifest.csv",
        "records": output / "per_clip.csv",
        "summary": output / "summary.csv",
        "metadata": output / "metadata.json",
    }

    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Outputs already exist: {existing}")

    write_csv(
        outputs["manifest"],
        ["clip_path", "sha256"],
        [
            {
                "clip_path": record["clip_path"],
                "sha256": record["sha256"],
            }
            for record in records
        ],
    )

    write_csv(
        outputs["records"],
        list(records[0]),
        records,
    )

    write_csv(
        outputs["summary"],
        list(summary[0]),
        summary,
    )

    with outputs["metadata"].open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "repository": REPO,
                "revision": REVISION,
                "seed": SEED,
                "splits": args.splits,
                "per_split": args.per_split,
            },
            stream,
            indent=2,
        )

    print(f"\nRevision: {REVISION}")

    for row in summary:
        print(
            f"{row['split']}: "
            f"{row['sampled_files']} sampled, "
            f"{row['eligible_shot_clips']} eligible, "
            f"{row['affected_clips']} affected, "
            f"{row['complete_candidate_loss_clips']} "
            "complete candidate-loss cases"
        )

    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
