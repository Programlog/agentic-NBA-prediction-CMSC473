"""Download selected BasketEvent review videos from Hugging Face."""

import argparse
import csv
from pathlib import Path, PurePosixPath

import cv2
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    downloaded = 0
    existing = 0
    failed = []

    for index, row in enumerate(rows, start=1):
        filename = row["video_path"].replace("\\", "/")
        relative = PurePosixPath(filename)

        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".mp4":
            raise ValueError(f"Unsafe video path: {filename}")

        destination = args.video_root.joinpath(*relative.parts)

        if destination.exists():
            existing += 1
            print(f"[{index}/{len(rows)}] Already exists: {filename}")
            continue

        print(f"[{index}/{len(rows)}] Downloading: {filename}")

        try:
            hf_hub_download(
                repo_id="zaywas/BasketEvent",
                repo_type="dataset",
                filename=filename,
                local_dir=str(args.video_root),
            )
            downloaded += 1
        except Exception as exc:
            failed.append((filename, str(exc)))
            print(f"  Download failed: {exc}")

    print("\n=== Download summary ===")
    print(f"Existing: {existing}")
    print(f"Downloaded: {downloaded}")
    print(f"Failed: {len(failed)}")

    readable = 0
    for row in rows:
        relative = PurePosixPath(row["video_path"].replace("\\", "/"))
        destination = args.video_root.joinpath(*relative.parts)

        if not destination.is_file():
            continue

        capture = cv2.VideoCapture(str(destination))
        try:
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    readable += 1
        finally:
            capture.release()

    print(f"Readable videos: {readable}/{len(rows)}")

    if failed or readable != len(rows):
        raise SystemExit("Some videos failed download/readability checks.")


if __name__ == "__main__":
    main()
