"""Select a reproducible exploratory BasketEvent release-review sample."""

import argparse
import csv
import random
from pathlib import Path

CATEGORIES = {
    "multiple_candidate_episodes": 8,
    "single_candidate_episode": 8,
    "release_not_found": 4,
}


def select_sample(report: Path, video_root: Path, seed: int = 473) -> list[dict]:
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    rng = random.Random(seed)
    selected = []

    for category, target in CATEGORIES.items():
        eligible = sorted(
            (
                row
                for row in rows
                if row["status"] == category and row["clip_path"].lower().endswith(".json")
            ),
            key=lambda row: row["clip_path"],
        )

        if len(eligible) < target:
            raise ValueError(f"{category}: need {target} clips, found {len(eligible)}")

        local = [
            row
            for row in eligible
            if (video_root / Path(row["clip_path"]).with_suffix(".mp4")).exists()
        ]

        remote = [
            row
            for row in eligible
            if not (video_root / Path(row["clip_path"]).with_suffix(".mp4")).exists()
        ]

        # Retain locally available footage, sampling when oversubscribed.
        chosen_local = rng.sample(local, min(len(local), target))

        needed = target - len(chosen_local)
        chosen_remote = rng.sample(remote, needed)

        for row in chosen_local + chosen_remote:
            video_path = Path(row["clip_path"]).with_suffix(".mp4")

            selected.append(
                {
                    "clip_path": row["clip_path"],
                    "video_path": video_path.as_posix(),
                    "diagnostic_status": category,
                    "selected_release_frame": row["selected_release_frame"],
                    "local_video_available": (video_root / video_path).exists(),
                }
            )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=473)
    args = parser.parse_args()

    sample = select_sample(args.report, args.video_root, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(sample[0]),
        )
        writer.writeheader()
        writer.writerows(sample)

    print(f"Selected {len(sample)} clips")
    print(
        "Already downloaded:",
        sum(row["local_video_available"] for row in sample),
    )


if __name__ == "__main__":
    main()
