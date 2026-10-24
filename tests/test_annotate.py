import json
import urllib.error
import urllib.request
from http.server import HTTPServer
from threading import Thread

import pandas as pd
import pytest

from nba_shots.annotate import (
    AnnotationRequestHandler,
    resolve_clip_path,
    update_manifest_row,
    validate_box,
)
from nba_shots.vision import load_manifest


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
    df = pd.DataFrame([
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
    ])
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


def test_annotation_http_server_endpoints(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    df = pd.DataFrame([
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
    ])
    df.to_csv(manifest_path, index=False)

    class CustomHandler(AnnotationRequestHandler):
        pass

    CustomHandler.manifest_path = manifest_path
    CustomHandler.video_dir = tmp_path
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

        # 3. POST /api/save updates manifest
        save_payload = json.dumps({
            "shot_id": "nba:001:10",
            "release_frame": 42,
            "shooter_box": [50, 60, 150, 200],
            "defender_box": None,
            "shooting_side": "left",
        }).encode("utf-8")

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

    finally:
        server.shutdown()
        server.server_close()
