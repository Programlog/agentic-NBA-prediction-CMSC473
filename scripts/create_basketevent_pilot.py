
from pathlib import Path
import pandas as pd

SHOT_ID = "nba:0022500003:582"

video = next(
    Path("data/basketevent/test/0022500003").glob("*.mp4")
)

shots = pd.read_parquet("data/recent/shots.parquet")
matched = shots[shots["shot_id"] == SHOT_ID]

if len(matched) != 1:
    raise ValueError(
        f"Expected one matching shot, found {len(matched)}"
    )

shot = matched.iloc[0]

manifest = pd.DataFrame([{
    "shot_id": SHOT_ID,
    "game_id": str(shot["game_id"]),
    "game_event_id": str(shot["game_event_id"]),
    "clip_path": str(video.resolve()),
    "release_frame": "",
    "shooter_box": "",
    "defender_box": "",
    "shooting_side": "",
}])

output = Path("data/basketevent/pilot_manifest.csv")
manifest.to_csv(output, index=False)

print("Created:", output)
print("Shot:", SHOT_ID)
print("Video:", video)
