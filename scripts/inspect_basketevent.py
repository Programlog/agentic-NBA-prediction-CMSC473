
import json
from pathlib import Path

root = Path("data/basketevent")
videos = sorted(root.rglob("*.mp4"))

if not videos:
    raise FileNotFoundError("No downloaded videos found.")

video = videos[0]
annotation = video.with_suffix(".json")

print("Video:", video)
print("Game ID:", video.parent.name)

with open(annotation, encoding="utf-8") as f:
    data = json.load(f)

print("\nAnnotation structure:", type(data).__name__)

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys())[:20])

    for entity_id, info in data.items():
        if not isinstance(info, dict):
            continue

        print("\nEntity:", entity_id)
        print("Player:", info.get("player_name"))
        print("Jersey:", info.get("jersey_number"))
        print("Event:", info.get("event"))
        print("Tracked frames:", len(info.get("trajectory") or []))
