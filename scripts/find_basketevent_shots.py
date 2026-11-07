
import csv
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "zaywas/BasketEvent"
DEST = Path("data/basketevent")

# Download 10 additional shooting clips.
TARGET_SHOTS = 10

# Skip the clip we already annotated.
SKIP_VIDEOS = {
    "test/0022500003/02336878-36b3-dd51-66e6-88d8103bbd1e_1280x720.mp4"
}

METADATA_FILE = DEST / "shot_candidates.csv"

SHOT_TYPES = {
    "made shot",
    "missed shot",
    "2pt field goal",
    "3pt field goal",
}

api = HfApi()

print("Listing BasketEvent files...", flush=True)

files = set(
    api.list_repo_files(
        repo_id=REPO,
        repo_type="dataset"
    )
)

videos = sorted(
    f for f in files if f.lower().endswith(".mp4")
)

print(f"Found {len(videos)} video files", flush=True)

# Preserve candidates found during previous runs.
fieldnames = [
    "game_id",
    "video_path",
    "json_path",
    "player",
    "period",
    "clock",
    "action_type",
    "description",
]

candidates = {}

if METADATA_FILE.exists():
    with METADATA_FILE.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            candidates[row["video_path"]] = row

print(
    f"Previously recorded candidates: {len(candidates)}",
    flush=True
)

new_downloads = 0
made_count = 0
missed_count = 0

for index, video_path in enumerate(videos, start=1):

    if new_downloads >= TARGET_SHOTS:
        break

    if video_path in SKIP_VIDEOS:
        continue

    if video_path in candidates:
        continue

    json_path = video_path[:-4] + ".json"

    if json_path not in files:
        continue

    print(
        f"\nChecking {index}/{len(videos)}: {json_path}",
        flush=True
    )

    try:
        local_json = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=json_path,
            local_dir=DEST
        )

        with open(local_json, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        shot_info = None

        # Find a shooting event in this video's annotations.
        for entity_id, info in data.items():

            if not isinstance(info, dict):
                continue

            event = info.get("event") or {}

            if not isinstance(event, dict):
                continue

            action = str(
                event.get("actionType", "")
            ).strip().lower()

            if action in SHOT_TYPES or "shot" in action:
                shot_info = info
                break

        if shot_info is None:
            continue

        event = shot_info["event"]

        action_type = str(
            event.get("actionType", "")
        )

        print("\nFOUND SHOOTING EVENT!")
        print("Video:", video_path)
        print("Player:", shot_info.get("player_name"))
        print("Action:", action_type)
        print("Description:", event.get("description"))

        # Download the corresponding MP4.
        local_video = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=video_path,
            local_dir=DEST
        )

        # Game ID comes from the dataset directory.
        game_id = Path(video_path).parent.name

        candidate = {
            "game_id": game_id,
            "video_path": video_path,
            "json_path": json_path,
            "player": (
                shot_info.get("player_name")
                or event.get("playerName")
                or ""
            ),
            "period": event.get("period", ""),
            "clock": event.get("clock", ""),
            "action_type": action_type,
            "description": event.get("description", ""),
        }

        candidates[video_path] = candidate

        # Save after each successful download so a
        # failed or interrupted run doesn't lose progress.
        METADATA_FILE.parent.mkdir(
            parents=True, exist_ok=True
        )

        with METADATA_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(candidates.values())

        new_downloads += 1

        if action_type.lower() == "made shot":
            made_count += 1
        elif action_type.lower() == "missed shot":
            missed_count += 1

        print("Downloaded video:", local_video)
        print(
            f"New clips: {new_downloads}/{TARGET_SHOTS}"
        )

        time.sleep(0.5)

    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        Exception
    ) as exc:
        print("Error checking file:", exc)

print("\n========== SUMMARY ==========")
print("New clips downloaded:", new_downloads)
print("Made shots:", made_count)
print("Missed shots:", missed_count)
print("Total recorded candidates:", len(candidates))
print("Metadata saved to:", METADATA_FILE)
