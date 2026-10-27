import base64
import json
import sys
import urllib.error
import urllib.request
from http.server import HTTPServer
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nba_shots.annotate import (
    AnnotationRequestHandler,
    frame_preview,
    resolve_clip_path,
    update_manifest_row,
    validate_box,
    video_metadata,
)
from nba_shots.vision import extract_cv, load_manifest


def write_clip(path, fps=60, frames=120):
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((64, 64, 3), index, dtype=np.uint8))
    finally:
        writer.release()


@pytest.mark.parametrize("fps", [24, 29.97, 60])
def test_metadata_and_preview_use_actual_clip_frames(tmp_path, fps):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.mp4"
    write_clip(path, fps)
    metadata = video_metadata(path)
    assert metadata["fps"] == pytest.approx(fps, abs=0.01)
    assert metadata["frame_count"] == 120
    index = round(fps)
    preview = frame_preview(path, index)
    image = cv2.imdecode(
        np.frombuffer(base64.b64decode(preview["jpeg_data_url"].split(",")[1]), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert preview["frame_index"] == index
    assert preview["width"] == preview["height"] == 64
    # Each synthetic frame has its index as its pixel intensity (allow codec loss).
    assert float(image.mean()) == pytest.approx(index, abs=5)
    with pytest.raises(ValueError):
        frame_preview(path, 120)


def test_metadata_rejects_missing_video(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="cannot be opened"):
        video_metadata(tmp_path / "missing.mp4")


def test_validate_box_formats_and_errors():
    assert validate_box("[10, 20, 50, 80]") == [10.0, 20.0, 50.0, 80.0]
    assert validate_box([10, 20, 50, 80]) == [10.0, 20.0, 50.0, 80.0]
    assert validate_box("10.5, 20.5, 50.5, 80.5") == [10.5, 20.5, 50.5, 80.5]
    assert validate_box(None) is None
    assert validate_box("") is None
    assert validate_box("null") is None

    with pytest.raises(ValueError, match="positive area"):
        validate_box([50, 80, 10, 20])
    with pytest.raises(ValueError, match="positive area"):
        validate_box([10, 20, 10, 80])
    with pytest.raises(ValueError, match="positive area"):
        validate_box([10, 20, 50])


def test_resolve_clip_path_auto_discovery(tmp_path):
    manifest_dir = tmp_path / "manifest_dir"
    manifest_dir.mkdir()
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    # Case 1: clip_path exists directly
    clip_file = manifest_dir / "clip1.mp4"
    clip_file.write_bytes(b"video-bytes")
    row1 = {"shot_id": "nba:1:1", "clip_path": str(clip_file)}
    assert resolve_clip_path(row1, manifest_dir, video_dir) == clip_file

    # Case 2: clip_path is relative
    row2 = {"shot_id": "nba:1:2", "clip_path": "clip1.mp4"}
    assert resolve_clip_path(row2, manifest_dir, video_dir) == clip_file

    # Case 3: clip_path is empty, discovered via shot_id in video_dir
    discovered = video_dir / "nba_1_3.mp4"
    discovered.write_bytes(b"video-bytes-3")
    row3 = {"shot_id": "nba:1:3", "clip_path": ""}
    assert resolve_clip_path(row3, manifest_dir, video_dir) == discovered

    # Case 4: Not found
    row4 = {"shot_id": "nba:1:99", "clip_path": ""}
    assert resolve_clip_path(row4, manifest_dir, video_dir) is None


def test_update_manifest_row_persistence(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    df = pd.DataFrame(
        [
            {
                "shot_id": "nba:001:10",
                "game_id": "001",
                "game_event_id": "10",
                "clip_path": "",
                "release_frame": "",
                "shooter_box": "",
                "defender_box": "",
                "shooting_side": "",
            }
        ]
    )
    df.to_csv(manifest_path, index=False)

    update_manifest_row(
        manifest_path=manifest_path,
        shot_id="nba:001:10",
        release_frame=35,
        shooter_box=[100, 150, 250, 400],
        shooting_side="right",
        defender_box=[260, 140, 380, 410],
        clip_path="videos/clip10.mp4",
    )

    reloaded = load_manifest(manifest_path)
    row = reloaded.iloc[0]
    assert int(row["release_frame"]) == 35
    assert json.loads(row["shooter_box"]) == [100.0, 150.0, 250.0, 400.0]
    assert json.loads(row["defender_box"]) == [260.0, 140.0, 380.0, 410.0]
    assert row["shooting_side"] == "right"
    assert row["clip_path"] == "videos/clip10.mp4"

    # Validation errors
    with pytest.raises(ValueError, match="non-negative"):
        update_manifest_row(manifest_path, "nba:001:10", -1, [10, 10, 20, 20], "right")
    with pytest.raises(ValueError, match="left"):
        update_manifest_row(manifest_path, "nba:001:10", 35, [10, 10, 20, 20], "center")
    with pytest.raises(ValueError, match="not found"):
        update_manifest_row(manifest_path, "nba:999:99", 35, [10, 10, 20, 20], "right")
    for invalid in [1.5, True]:
        with pytest.raises(ValueError, match="integer"):
            update_manifest_row(manifest_path, "nba:001:10", invalid, [10, 10, 20, 20], "right")


def test_annotation_http_server_endpoints(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.csv"
    df = pd.DataFrame(
        [
            {
                "shot_id": "nba:001:10",
                "game_id": "001",
                "game_event_id": "10",
                "clip_path": "",
                "release_frame": "",
                "shooter_box": "",
                "defender_box": "",
                "shooting_side": "",
            }
        ]
    )
    df.to_csv(manifest_path, index=False)
    clips = tmp_path / "downloaded"
    clips.mkdir()
    clip = clips / "nba_001_10.mp4"
    write_clip(clip)

    class CustomHandler(AnnotationRequestHandler):
        pass

    CustomHandler.manifest_path = manifest_path
    CustomHandler.video_dir = clips
    CustomHandler.weights = "yolo11n-pose.pt"
    CustomHandler.device = "cpu"

    server = HTTPServer(("127.0.0.1", 0), CustomHandler)
    port = server.server_port
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{port}"

        # 1. GET / serves HTML
        with urllib.request.urlopen(f"{base_url}/") as res:
            assert res.status == 200
            html = res.read().decode("utf-8")
            assert "NBA Shot Video Annotation Tool" in html

        # 2. GET /api/shots serves JSON
        with urllib.request.urlopen(f"{base_url}/api/shots") as res:
            assert res.status == 200
            shots = json.loads(res.read().decode("utf-8"))
            assert len(shots) == 1
            assert shots[0]["shot_id"] == "nba:001:10"

        with urllib.request.urlopen(f"{base_url}/api/video-info?shot_index=0") as res:
            metadata = json.load(res)
            assert metadata == {"fps": 60.0, "frame_count": 120}
        with urllib.request.urlopen(f"{base_url}/api/frame?shot_index=0&frame_index=42") as res:
            assert json.load(res)["frame_index"] == 42
        for suffix in ["video-info?shot_index=-1", "frame?shot_index=0&frame_index=120"]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(f"{base_url}/api/{suffix}")
            assert error.value.code == 400

        # 3. POST /api/save updates manifest
        save_payload = json.dumps(
            {
                "shot_id": "nba:001:10",
                "release_frame": 42,
                "shooter_box": [0, 0, 32, 64],
                "defender_box": None,
                "shooting_side": "left",
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/save",
            data=save_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            body = json.loads(res.read().decode("utf-8"))
            assert body.get("ok") is True

        reloaded = pd.read_csv(manifest_path)
        assert int(reloaded.iloc[0]["release_frame"]) == 42
        assert reloaded.iloc[0]["shooting_side"] == "left"
        assert reloaded.iloc[0]["clip_path"] == "downloaded/nba_001_10.mp4"
        with urllib.request.urlopen(f"{base_url}/api/shots") as res:
            reopened = json.load(res)[0]
            assert reopened["release_frame"] == 42
            assert reopened["clip_path"] == "downloaded/nba_001_10.mp4"
        for invalid_frame in [-1, 120, 42.5, True]:
            invalid_payload = json.loads(save_payload)
            invalid_payload["release_frame"] = invalid_frame
            invalid_request = urllib.request.Request(
                f"{base_url}/api/save",
                data=json.dumps(invalid_payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(invalid_request)
            assert error.value.code == 400
        assert int(load_manifest(manifest_path).iloc[0].release_frame) == 42

        # Real decoding and manifest handoff, with only pose inference substituted.
        # No model weights or external downloads are needed for this regression.
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Model:
            ckpt_path = str(clip)

            def predict(self, image, **kwargs):
                assert float(image.mean()) == pytest.approx(42, abs=5)
                pose = np.ones((1, 17, 3))
                pose[0, 5, :2], pose[0, 6, :2] = [10, 10], [20, 10]
                pose[0, 11, :2], pose[0, 12, :2] = [10, 30], [20, 30]
                return [
                    SimpleNamespace(
                        boxes=SimpleNamespace(xyxy=Array([[0, 0, 32, 64]])),
                        keypoints=SimpleNamespace(data=Array(pose)),
                        plot=lambda: image,
                    )
                ]

        monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: Model()))
        elsewhere = tmp_path / "other-working-directory"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        output = tmp_path / "extraction"
        extract_cv(manifest_path, output, weights="fake-test-weights")
        features = pd.read_parquet(output / "cv_features.parquet").iloc[0]
        assert features["cv_status"] == "partial"  # No defender annotated.
        assert features["latest_frame_used"] == 42
        assert pd.isna(features["cv_error"])

    finally:
        server.shutdown()
        server.server_close()
