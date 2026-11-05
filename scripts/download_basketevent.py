
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "zaywas/BasketEvent"
DEST = Path("data/basketevent")

api = HfApi()

print("Listing BasketEvent dataset files...")
files = api.list_repo_files(
    repo_id=REPO,
    repo_type="dataset"
)

videos = sorted(
    f for f in files if f.lower().endswith(".mp4")
)

print(f"Found {len(videos)} MP4 files.")

if not videos:
    raise RuntimeError(
        "No MP4 files found. Inspect the dataset layout "
        "before attempting to download."
    )

# Prefer a clip for a game previously used in our NBA project.
preferred_game = "0022500004"

candidates = [
    f for f in videos if preferred_game in f.split("/")
]

if not candidates:
    print("Preferred game unavailable; using first sample.")
    candidates = videos

video_path = candidates[0]
json_path = video_path[:-4] + ".json"

print("Selected video:", video_path)

for path in [video_path, json_path]:
    if path not in files:
        print("No matching annotation:", path)
        continue

    local = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=path,
        local_dir=DEST
    )
    print("Downloaded:", local)

print("Finished.")
