
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "zaywas/BasketEvent"
DEST = Path("data/basketevent")

api = HfApi()

print("Listing BasketEvent files...", flush=True)

files = api.list_repo_files(
    repo_id=REPO,
    repo_type="dataset"
)

videos = sorted(
    f for f in files if f.lower().endswith(".mp4")
)

print(f"Found {len(videos)} video files", flush=True)

shot_types = {
    "made shot",
    "missed shot",
    "2pt field goal",
    "3pt field goal",
}

for index, video_path in enumerate(videos, start=1):
    json_path = video_path[:-4] + ".json"

    if json_path not in files:
        continue

    print(
        f"Checking {index}/{len(videos)}: {json_path}",
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

        for entity_id, info in data.items():
            if not isinstance(info, dict):
                continue

            event = info.get("event") or {}
            action = str(event.get("actionType", "")).lower()

            if action in shot_types or "shot" in action:
                print("\nFOUND SHOOTING EVENT!")
                print("Video:", video_path)
                print("Player:", info.get("player_name"))
                print("Event:", event)

                local_video = hf_hub_download(
                    repo_id=REPO,
                    repo_type="dataset",
                    filename=video_path,
                    local_dir=DEST
                )

                print("Downloaded video:", local_video)
                raise SystemExit(0)

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("Error checking file:", exc)

print("No shooting event found.")
