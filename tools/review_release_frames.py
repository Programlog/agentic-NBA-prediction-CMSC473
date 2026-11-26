"""Frame-by-frame viewer for manually reviewing basketball shot releases."""

import argparse
from pathlib import Path

import cv2


def review_video(video_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if frame_count <= 0:
        capture.release()
        raise RuntimeError("Video contains no readable frames")

    print(f"Video: {video_path}")
    print(f"Frames: {frame_count}, FPS: {fps:.2f}")
    print("Controls: D = next | A = previous | F = +10 | B = -10")
    print("          P = print frame number | Q = quit")

    frame_index = 0

    try:
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()

            if not success:
                print(f"Unable to read frame {frame_index}")
                break

            height, width = frame.shape[:2]
            scale = min(1.0, 1100 / width, 700 / height)

            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.putText(
                frame,
                f"Frame {frame_index}/{frame_count - 1}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow("NBA Release Frame Review", frame)

            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                frame_index = min(frame_index + 1, frame_count - 1)
            elif key == ord("a"):
                frame_index = max(frame_index - 1, 0)
            elif key == ord("f"):
                frame_index = min(frame_index + 10, frame_count - 1)
            elif key == ord("b"):
                frame_index = max(frame_index - 10, 0)
            elif key == ord("p"):
                print(f"Selected frame: {frame_index}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    args = parser.parse_args()

    review_video(args.video)


if __name__ == "__main__":
    main()
