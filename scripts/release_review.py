"""Blind review of the automatic BasketEvent release frames on a random sample of video shots.

    python scripts/release_review.py make   # sample, download clips, draw strips, blank CSV
    python scripts/release_review.py score  # compare the filled-in CSV with the detector

The reviewer sees only review.csv and the images. Strips are centred a random few frames
away from the detected release, so the middle frame does not give the answer away; the
detected frames live in answers.csv, which the reviewer should not open.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nba_shots.basketevent import REPO, SHOT_ACTIONS, boxes

COHORT = Path("data/processed/basketevent_cohort_v2")
LABELS = Path("data/basketevent_labels_v2")
OUT = Path("data/release_review")
SAMPLE, SEED = 50, 42
HALF, JITTER = 7, 3  # 15 frames per strip, centred up to 3 frames off the detected release
CROP, TILE = 480, 256  # crop around the shooter, then shrink each frame to a tile
GREEN, RED, YELLOW = (60, 200, 60), (40, 40, 230), (0, 220, 255)


INSTRUCTIONS = """Release frame review: one row of review.csv per shot, items 01-50.

For each item, open strips/NN_frames.jpg: 15 consecutive video frames, numbered, with the
shooter's box in green and the ball's in yellow (the boxes come from the dataset and can be
wrong). Find the release: the first frame where the ball has clearly left the shooter's
hands.

  true_release_frame  the number of the last frame where the ball still touches the hands.
                      If the release is not in the strip, play videos/NN.mp4 (frames
                      count from 0, 25 per second) and write the frame number; if you cannot
                      tell, leave it blank and say why in notes.
  shooter_correct     y/n: in strips/NN_players.jpg, is the green box the player shooting?
  defender_correct    y/n: is the red box the defender nearest to the shooter (n if no red
                      box but a defender is close)?

Do not open answers.csv. Send back review.csv when done.
"""


def shooter_track(labels: dict, clip: str) -> tuple[np.ndarray, np.ndarray]:
    entities = labels[clip]
    key = next(
        k
        for k, v in entities.items()
        if isinstance(v, dict)
        and isinstance(v.get("event"), dict)
        and v["event"].get("actionType") in SHOT_ACTIONS
    )
    return boxes(entities[key]), boxes(entities.get("ball") or {})


def rectangle(image, box, colour, scale=1.0, origin=(0, 0)):
    if np.isfinite(box).all():
        x, y, w, h = box
        top = (int((x - origin[0]) * scale), int((y - origin[1]) * scale))
        bottom = (int((x + w - origin[0]) * scale), int((y + h - origin[1]) * scale))
        cv2.rectangle(image, top, bottom, colour, 2)


def make() -> None:
    from huggingface_hub import hf_hub_download

    if OUT.exists():
        raise SystemExit(f"{OUT} exists; delete it to draw a new sample")
    features = pd.read_parquet(COHORT / "box_features.parquet")
    shots = features.loc[features.has_video].sample(SAMPLE, random_state=SEED)
    revision = json.loads((LABELS / "source.json").read_text())["revision"]
    rng = np.random.default_rng(SEED)
    (OUT / "strips").mkdir(parents=True)
    (OUT / "videos").mkdir()
    review, answers = [], []
    for n, row in enumerate(shots.itertuples(), 1):
        game = row.clip.split("/")[1]
        with gzip.open(LABELS / "games" / f"{game}_labels.json.gz", "rt") as stream:
            labels = json.load(stream)
        video = hf_hub_download(
            REPO, row.clip.removesuffix(".json") + ".mp4", repo_type="dataset", revision=revision
        )
        shooter, ball = shooter_track(labels, row.clip)
        release = int(row.release_frame)
        centre = release + int(rng.integers(-JITTER, JITTER + 1))
        frames = range(max(0, centre - HALF), centre + HALF + 1)
        capture = cv2.VideoCapture(video)
        images = {}
        for k in range(max(frames) + 1):
            ok, image = capture.read()
            if not ok:
                break
            if k in frames or k == release:
                images[k] = image
        capture.release()

        # Crop every frame around the shooter's position at release so the hands stay in view.
        x, y, w, h = shooter[release]
        left = int(np.clip(x + w / 2 - CROP / 2, 0, 1280 - CROP))
        top = int(np.clip(y + h / 2 - CROP * 0.6, 0, 720 - CROP))
        scale = TILE / CROP
        tiles = []
        for k in frames:
            tile = np.zeros((TILE, TILE, 3), np.uint8)
            if k in images:
                tile = cv2.resize(images[k][top : top + CROP, left : left + CROP], (TILE, TILE))
                rectangle(
                    tile,
                    shooter[k] if k < len(shooter) else [np.nan] * 4,
                    GREEN,
                    scale,
                    (left, top),
                )
                rectangle(
                    tile, ball[k] if k < len(ball) else [np.nan] * 4, YELLOW, scale, (left, top)
                )
            cv2.putText(tile, str(k), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(tile)
        while len(tiles) % 5:
            tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
        strip = np.vstack([np.hstack(tiles[i : i + 5]) for i in range(0, len(tiles), 5)])
        name = f"{n:02d}"
        cv2.imwrite(str(OUT / "strips" / f"{name}_frames.jpg"), strip)

        # Full frame at release: is the green box the shooter and the red box their defender?
        whole = images[release].copy()
        rectangle(whole, shooter[release], GREEN)
        if isinstance(row.defender_box, str):
            x0, y0, x1, y1 = json.loads(row.defender_box)
            rectangle(whole, np.array([x0, y0, x1 - x0, y1 - y0]), RED)
        cv2.imwrite(str(OUT / "strips" / f"{name}_players.jpg"), whole)
        shutil.copy(video, OUT / "videos" / f"{name}.mp4")  # the pack travels as one folder

        review.append(
            {
                "item": name,
                "first_frame": min(frames),
                "last_frame": max(frames),
                "video": f"videos/{name}.mp4",
                "true_release_frame": "",
                "shooter_correct": "",
                "defender_correct": "",
                "notes": "",
            }
        )
        answers.append(
            {"item": name, "shot_id": row.shot_id, "clip": row.clip, "release_frame": release}
        )
        print(f"{name}: {row.clip}", flush=True)
    pd.DataFrame(review).to_csv(OUT / "review.csv", index=False)
    pd.DataFrame(answers).to_csv(OUT / "answers.csv", index=False)
    (OUT / "README.txt").write_text(INSTRUCTIONS)
    print(f"Wrote {len(review)} items to {OUT}")


def score() -> None:
    review = pd.read_csv(OUT / "review.csv").merge(pd.read_csv(OUT / "answers.csv"), on="item")
    marked = pd.to_numeric(review.true_release_frame, errors="coerce")
    error = (review.release_frame - marked).dropna()
    yes = {"y", "yes", "1", "true"}
    result = {
        "reviewed": int(marked.notna().sum()),
        "within_1_frame": float((error.abs() <= 1).mean()),
        "within_3_frames": float((error.abs() <= 3).mean()),
        "median_abs_error_frames": float(error.abs().median()),
        "mean_error_frames": float(error.mean()),  # positive: detector is late
        "shooter_correct": float(review.shooter_correct.dropna().astype(str).str.lower().isin(yes).mean()),
        "defender_correct": float(review.defender_correct.dropna().astype(str).str.lower().isin(yes).mean()),
    }
    (OUT / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
