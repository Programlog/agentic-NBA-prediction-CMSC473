"""Render manually chosen review windows; this helper never reads detector outputs."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from nba_shots.release_audit import render_frames


def details(index, start, stop, crop, audit="data/release_audit_50"):
    root = Path(audit)
    rows = json.loads((root / "blind/review_index.json").read_text())
    shot = rows[index]["shot_id"]
    render_frames(root, shot, start, stop, 1)
    folder = root / "blind/frames" / shot.replace(":", "_")
    tile_width, tile_height = 300, 350
    sheet = Image.new("RGB", (1200, tile_height * math.ceil((stop - start + 1) / 4)), "#dddddd")
    for k, frame in enumerate(range(start, stop + 1)):
        image = Image.open(folder / f"frame_{frame:05d}.png").crop(crop)
        image.thumbnail((tile_width, tile_height - 26))
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(image, (0, 26))
        ImageDraw.Draw(tile).text((5, 5), f"index {index} | frame {frame}", fill="black")
        sheet.paste(tile, ((k % 4) * tile_width, (k // 4) * tile_height))
    output = folder / f"detail_{start}_{stop}.jpg"
    sheet.save(output, quality=98)
    with (root / "blind/windows.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "index": index,
                    "shot_id": shot,
                    "start": start,
                    "stop": stop,
                    "crop": crop,
                    "sheet": str(output),
                }
            )
            + "\n"
        )
    print(json.dumps({"index": index, "sheet": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, help="JSON list of [index,start,stop,crop]")
    args = parser.parse_args()
    for window in json.loads(args.windows):
        details(*window)
