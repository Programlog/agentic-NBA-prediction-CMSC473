"""Interactive video annotation tool for release frames and player bounding boxes."""

from __future__ import annotations

import base64
import json
import re
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .vision import read_release_frame


def resolve_clip_path(row: dict, manifest_dir: Path, video_dir: Path | None = None) -> Path | None:
    """Find the video clip file for a manifest row."""
    clip_val = row.get("clip_path")
    if pd.notna(clip_val) and str(clip_val).strip():
        raw_path = Path(str(clip_val).strip())
        if raw_path.is_file():
            return raw_path
        relative = manifest_dir / raw_path
        if relative.is_file():
            return relative

    shot_id = str(row.get("shot_id", "")).strip()
    if not shot_id:
        return None

    filename = shot_id.replace(":", "_") + ".mp4"
    search_dirs = [
        d
        for d in [video_dir, manifest_dir, manifest_dir / "videos", Path("data/recent/videos")]
        if d and d.is_dir()
    ]
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate

    return None


def validate_box(box_value: Any) -> list[float] | None:
    """Validate and normalize a bounding box [x1, y1, x2, y2]."""
    if box_value is None:
        return None
    if isinstance(box_value, (float, int, str)) and pd.isna(box_value):
        return None
    if isinstance(box_value, str):
        text = box_value.strip()
        if text in {"", "null", "None"}:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            # Handle comma-separated format "x1,y1,x2,y2"
            parts = [float(p.strip()) for p in text.strip("[]").split(",") if p.strip()]
            parsed = parts
    else:
        parsed = box_value

    arr = np.asarray(parsed, dtype=float)
    if arr.shape != (4,) or not np.isfinite(arr).all() or arr[2] <= arr[0] or arr[3] <= arr[1]:
        raise ValueError(
            f"Person box must be [x1, y1, x2, y2] with positive area, got: {box_value}"
        )
    return [round(float(v), 2) for v in arr]


def detect_frame_boxes(
    clip_path: str | Path,
    frame_index: int,
    weights: str = "yolo11n-pose.pt",
    device: str = "cpu",
) -> dict:
    """Extract a frame, run YOLO detection, and return image + detected person boxes."""
    import cv2
    from ultralytics import YOLO

    clip_path = Path(clip_path)
    if not clip_path.is_file():
        raise FileNotFoundError(f"Video clip not found: {clip_path}")

    frame = read_release_frame(clip_path, frame_index, cv2)
    h, w = frame.shape[:2]

    model = YOLO(weights)
    prediction = model.predict(frame, device=device, verbose=False)[0]

    detected_boxes = []
    if prediction.boxes is not None and len(prediction.boxes):
        xyxy = prediction.boxes.xyxy.cpu().numpy()
        confs = (
            prediction.boxes.conf.cpu().numpy()
            if prediction.boxes.conf is not None
            else np.ones(len(xyxy))
        )
        for box, conf in zip(xyxy, confs):
            detected_boxes.append(
                {
                    "box": [round(float(b), 2) for b in box],
                    "confidence": round(float(conf), 3),
                }
            )

    # Encode frame to JPEG
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Failed to encode frame to JPEG")
    jpeg_b64 = base64.b64encode(buffer).decode("ascii")

    return {
        "frame_index": frame_index,
        "width": w,
        "height": h,
        "boxes": detected_boxes,
        "jpeg_data_url": f"data:image/jpeg;base64,{jpeg_b64}",
    }


def update_manifest_row(
    manifest_path: str | Path,
    shot_id: str,
    release_frame: int,
    shooter_box: list[float] | str,
    shooting_side: str,
    defender_box: list[float] | str | None = None,
    clip_path: str | None = None,
) -> pd.DataFrame:
    """Safely update a single shot in manifest.csv and persist changes."""
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path, dtype={"shot_id": str, "game_id": str, "game_event_id": str})
    for col in ["clip_path", "shooter_box", "defender_box", "shooting_side"]:
        if col in df.columns:
            df[col] = df[col].astype(object)

    if shot_id not in df["shot_id"].values:
        raise ValueError(f"shot_id {shot_id!r} not found in manifest")

    if release_frame < 0:
        raise ValueError("release_frame must be a non-negative integer")
    if shooting_side not in {"left", "right"}:
        raise ValueError("shooting_side must be 'left' or 'right'")

    s_box = validate_box(shooter_box)
    if s_box is None:
        raise ValueError("shooter_box is required")
    d_box = validate_box(defender_box)

    idx = df.index[df["shot_id"] == shot_id][0]
    df.loc[idx, "release_frame"] = int(release_frame)
    df.loc[idx, "shooter_box"] = json.dumps(s_box)
    df.loc[idx, "shooting_side"] = shooting_side
    df.loc[idx, "defender_box"] = json.dumps(d_box) if d_box else ""
    if clip_path:
        df.loc[idx, "clip_path"] = str(clip_path)

    df.to_csv(manifest_path, index=False)
    return df


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NBA Shot Annotation Tool</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --success: #22c55e;
      --shooter: #3b82f6;
      --defender: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
        sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: var(--card);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 { font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
    header .status { font-size: 0.85rem; color: var(--muted); }
    .main-layout {
      display: flex;
      flex: 1;
      height: calc(100vh - 60px);
      overflow: hidden;
    }
    .sidebar {
      width: 320px;
      background: var(--card);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
    }
    .sidebar-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      font-weight: 600;
      font-size: 0.9rem;
      display: flex;
      justify-content: space-between;
    }
    .shot-list {
      flex: 1;
      overflow-y: auto;
      list-style: none;
    }
    .shot-item {
      padding: 10px 16px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
      transition: background 0.15s;
    }
    .shot-item:hover { background: rgba(56, 189, 248, 0.1); }
    .shot-item.active { background: rgba(56, 189, 248, 0.2); font-weight: 600; }
    .badge {
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
    }
    .badge.done { background: rgba(34, 197, 94, 0.2); color: var(--success); }
    .badge.pending { background: rgba(148, 163, 184, 0.2); color: var(--muted); }
    .content {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 16px;
      overflow-y: auto;
      gap: 16px;
    }
    .video-viewport {
      position: relative;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 480px;
    }
    video, canvas {
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      object-fit: contain;
    }
    canvas { pointer-events: auto; cursor: crosshair; }
    .controls {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .controls-row {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    button {
      background: #334155;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 0.85rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s;
    }
    button:hover { background: #475569; }
    button.primary { background: var(--accent); color: #000; font-weight: 600; border: none; }
    button.primary:hover { background: #7dd3fc; }
    button.success { background: var(--success); color: #000; font-weight: 600; border: none; }
    button.success:hover { background: #4ade80; }
    button.active-mode { outline: 2px solid var(--accent); }
    .slider-container {
      flex: 1;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    input[type=range] { flex: 1; cursor: pointer; }
    .form-group {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
    }
    select, input[type=text] {
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 0.85rem;
    }
    .shortcuts-help {
      font-size: 0.75rem;
      color: var(--muted);
      line-height: 1.4;
    }
    kbd {
      background: #334155;
      padding: 2px 4px;
      border-radius: 3px;
      font-family: monospace;
      color: var(--text);
    }
  </style>
</head>
<body>
  <header>
    <h1>🏀 NBA Shot Video Annotation Tool</h1>
    <div class="status" id="manifest-stats">Loading shots...</div>
  </header>
  <div class="main-layout">
    <div class="sidebar">
      <div class="sidebar-header">
        <span>Shot Queue</span>
        <span id="queue-count">0</span>
      </div>
      <ul class="shot-list" id="shot-list"></ul>
    </div>
    <div class="content">
      <div class="video-viewport" id="viewport">
        <video id="video-player" playsinline></video>
        <canvas id="overlay-canvas"></canvas>
      </div>
      <div class="controls">
        <div class="controls-row">
          <button id="btn-play-pause">▶ Play</button>
          <button id="btn-prev-frame">◀ -1 Frame</button>
          <button id="btn-next-frame">▶ +1 Frame</button>
          <div class="slider-container">
            <span id="frame-counter" style="font-family:monospace; min-width:80px;">Frame: 0</span>
            <input type="range" id="frame-slider" min="0" max="100" value="0">
          </div>
          <button id="btn-mark-release" class="primary">📌 Set Release Frame (R)</button>
        </div>
        <div class="controls-row" style="border-top: 1px solid var(--border); padding-top: 12px;">
          <button id="btn-run-yolo">🤖 Run YOLO on Frame</button>
          <button id="btn-assign-shooter" class="active-mode">🎯 Mode: Shooter Box</button>
          <button id="btn-assign-defender">🛡️ Mode: Defender Box</button>
          <div class="form-group">
            <label>Shooting Hand:</label>
            <select id="select-hand">
              <option value="right">Right</option>
              <option value="left">Left</option>
            </select>
          </div>
          <div class="form-group" style="margin-left: auto;">
            <button id="btn-clear-boxes">Clear Boxes</button>
            <button id="btn-save" class="success">💾 Save & Next (S)</button>
          </div>
        </div>
        <div class="controls-row shortcuts-help">
          <span>Shortcuts: <kbd>Space</kbd> Play/Pause &bull; <kbd>←</kbd>/<kbd>→</kbd> Prev/Next
            &bull; <kbd>R</kbd> Set Release &bull; <kbd>S</kbd> Save &bull; Click to assign</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    let manifestData = [];
    let currentShotIndex = 0;
    let detectedBoxes = [];
    let shooterBox = null;
    let defenderBox = null;
    let assignMode = 'shooter';
    const video = document.getElementById('video-player');
    const canvas = document.getElementById('overlay-canvas');
    const ctx = canvas.getContext('2d');
    const frameSlider = document.getElementById('frame-slider');
    const frameCounter = document.getElementById('frame-counter');

    async function loadShots() {
      const res = await fetch('/api/shots');
      manifestData = await res.json();
      renderShotList();
      if (manifestData.length > 0) {
        selectShot(0);
      }
    }

    function renderShotList() {
      const list = document.getElementById('shot-list');
      list.innerHTML = '';
      let doneCount = 0;
      manifestData.forEach((shot, idx) => {
        const isDone = shot.release_frame !== "" && shot.release_frame !== null && shot.shooter_box;
        if (isDone) doneCount++;
        const li = document.createElement('li');
        li.className = 'shot-item' + (idx === currentShotIndex ? ' active' : '');
        const badgeClass = isDone ? 'done' : 'pending';
        const badgeLabel = isDone ? 'Annotated' : 'Pending';
        li.innerHTML = `<span>${shot.shot_id}</span>` +
          `<span class="badge ${badgeClass}">${badgeLabel}</span>`;
        li.onclick = () => selectShot(idx);
        list.appendChild(li);
      });
      const pct = Math.round((doneCount / manifestData.length) * 100 || 0);
      document.getElementById('manifest-stats').innerText =
        `${doneCount} of ${manifestData.length} shots annotated (${pct}%)`;
      document.getElementById('queue-count').innerText = manifestData.length;
    }

    function selectShot(index) {
      currentShotIndex = index;
      renderShotList();
      const shot = manifestData[index];
      shooterBox = shot.shooter_box ? JSON.parse(shot.shooter_box) : null;
      defenderBox = shot.defender_box ? JSON.parse(shot.defender_box) : null;
      document.getElementById('select-hand').value = shot.shooting_side || 'right';
      detectedBoxes = [];

      video.src = `/api/video?shot_index=${index}`;
      video.load();
      video.onloadedmetadata = () => {
        resizeCanvas();
        frameSlider.max = Math.floor(video.duration * 30);
        if (shot.release_frame !== "" && shot.release_frame !== null) {
          seekToFrame(parseInt(shot.release_frame));
        } else {
          seekToFrame(0);
        }
      };
    }

    function resizeCanvas() {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 360;
      redrawOverlay();
    }

    function currentFrameIndex() {
      return Math.round(video.currentTime * 30);
    }

    function seekToFrame(frame) {
      video.pause();
      video.currentTime = Math.max(0, frame / 30);
      frameSlider.value = frame;
      frameCounter.innerText = `Frame: ${frame}`;
      redrawOverlay();
    }

    video.ontimeupdate = () => {
      const f = currentFrameIndex();
      frameSlider.value = f;
      frameCounter.innerText = `Frame: ${f}`;
      redrawOverlay();
    };

    frameSlider.oninput = (e) => seekToFrame(parseInt(e.target.value));

    document.getElementById('btn-play-pause').onclick = () => {
      if (video.paused) { video.play(); } else { video.pause(); }
    };
    document.getElementById('btn-prev-frame').onclick = () => seekToFrame(currentFrameIndex() - 1);
    document.getElementById('btn-next-frame').onclick = () => seekToFrame(currentFrameIndex() + 1);

    document.getElementById('btn-mark-release').onclick = () => {
      const f = currentFrameIndex();
      manifestData[currentShotIndex].release_frame = f;
      alert(`Release frame set to ${f}. Running YOLO person detection...`);
      runYolo(f);
    };

    async function runYolo(frameIdx) {
      const f = frameIdx !== undefined ? frameIdx : currentFrameIndex();
      const res = await fetch('/api/detect-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shot_index: currentShotIndex, frame_index: f })
      });
      const data = await res.json();
      if (data.error) {
        alert('YOLO error: ' + data.error);
        return;
      }
      detectedBoxes = data.boxes.map(b => b.box);
      redrawOverlay();
    }

    document.getElementById('btn-run-yolo').onclick = () => runYolo();

    document.getElementById('btn-assign-shooter').onclick = () => {
      assignMode = 'shooter';
      document.getElementById('btn-assign-shooter').className = 'active-mode';
      document.getElementById('btn-assign-defender').className = '';
    };

    document.getElementById('btn-assign-defender').onclick = () => {
      assignMode = 'defender';
      document.getElementById('btn-assign-defender').className = 'active-mode';
      document.getElementById('btn-assign-shooter').className = '';
    };

    document.getElementById('btn-clear-boxes').onclick = () => {
      shooterBox = null;
      defenderBox = null;
      redrawOverlay();
    };

    canvas.onclick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      for (const box of detectedBoxes) {
        if (x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3]) {
          if (assignMode === 'shooter') {
            shooterBox = box;
          } else {
            defenderBox = box;
          }
          redrawOverlay();
          return;
        }
      }
    };

    function redrawOverlay() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      detectedBoxes.forEach((box, i) => {
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 2;
        ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans-serif';
        ctx.fillText(`Person ${i + 1}`, box[0] + 4, box[1] + 16);
      });

      if (shooterBox) {
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 4;
        ctx.strokeRect(
          shooterBox[0], shooterBox[1],
          shooterBox[2] - shooterBox[0], shooterBox[3] - shooterBox[1]
        );
        ctx.fillStyle = '#3b82f6';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('🎯 SHOOTER', shooterBox[0] + 4, shooterBox[1] + 20);
      }

      if (defenderBox) {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 4;
        ctx.strokeRect(
          defenderBox[0], defenderBox[1],
          defenderBox[2] - defenderBox[0], defenderBox[3] - defenderBox[1]
        );
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('🛡️ DEFENDER', defenderBox[0] + 4, defenderBox[1] + 20);
      }
    }

    async function saveCurrent() {
      const relFrame = manifestData[currentShotIndex].release_frame;
      if (relFrame === "" || relFrame === null || relFrame === undefined) {
        alert('Please mark the release frame first (Press R or click Set Release Frame).');
        return;
      }
      if (!shooterBox) {
        alert('Please assign a Shooter box before saving.');
        return;
      }
      const payload = {
        shot_id: manifestData[currentShotIndex].shot_id,
        release_frame: parseInt(relFrame),
        shooter_box: shooterBox,
        defender_box: defenderBox,
        shooting_side: document.getElementById('select-hand').value
      };
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.ok) {
        manifestData[currentShotIndex].release_frame = payload.release_frame;
        manifestData[currentShotIndex].shooter_box = JSON.stringify(payload.shooter_box);
        const defJson = payload.defender_box ? JSON.stringify(payload.defender_box) : "";
        manifestData[currentShotIndex].defender_box = defJson;
        manifestData[currentShotIndex].shooting_side = payload.shooting_side;
        renderShotList();
        if (currentShotIndex + 1 < manifestData.length) {
          selectShot(currentShotIndex + 1);
        } else {
          alert('All shots annotated!');
        }
      } else {
        alert('Save error: ' + data.error);
      }
    }

    document.getElementById('btn-save').onclick = saveCurrent;

    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') {
        e.preventDefault();
        document.getElementById('btn-play-pause').click();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        document.getElementById('btn-prev-frame').click();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        document.getElementById('btn-next-frame').click();
      } else if (e.key === 'r' || e.key === 'R') {
        document.getElementById('btn-mark-release').click();
      } else if (e.key === 's' || e.key === 'S') {
        saveCurrent();
      }
    });

    loadShots();
  </script>
</body>
</html>
"""


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    manifest_path: Path
    video_dir: Path | None
    weights: str
    device: str

    def log_message(self, format, *args):
        # Suppress noisy HTTP logs during interactive usage
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in {"/", "/index.html"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if path == "/api/shots":
            df = pd.read_csv(
                self.manifest_path,
                dtype={"shot_id": str, "game_id": str, "game_event_id": str},
            )
            records = df.to_dict("records")
            clean_records = []
            for r in records:
                clean_records.append({k: ("" if pd.isna(v) else v) for k, v in r.items()})
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(clean_records).encode("utf-8"))
            return

        if path == "/api/video":
            shot_idx_list = query.get("shot_index")
            if not shot_idx_list:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing shot_index")
                return
            shot_idx = int(shot_idx_list[0])
            df = pd.read_csv(
                self.manifest_path,
                dtype={"shot_id": str, "game_id": str, "game_event_id": str},
            )
            if shot_idx < 0 or shot_idx >= len(df):
                self.send_error(HTTPStatus.NOT_FOUND, "Shot index out of range")
                return
            row = df.iloc[shot_idx].to_dict()
            clip = resolve_clip_path(row, self.manifest_path.parent, self.video_dir)
            if not clip or not clip.is_file():
                self.send_error(
                    HTTPStatus.NOT_FOUND,
                    f"Video clip not found for shot {row.get('shot_id')}",
                )
                return

            self.serve_file(clip)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

        if path == "/api/detect-frame":
            shot_idx = body.get("shot_index", 0)
            frame_idx = body.get("frame_index", 0)
            df = pd.read_csv(
                self.manifest_path,
                dtype={"shot_id": str, "game_id": str, "game_event_id": str},
            )
            row = df.iloc[shot_idx].to_dict()
            clip = resolve_clip_path(row, self.manifest_path.parent, self.video_dir)
            if not clip or not clip.is_file():
                self.send_json({"error": "Clip file not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                result = detect_frame_boxes(clip, frame_idx, self.weights, self.device)
                self.send_json(result)
            except Exception as e:
                self.send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/save":
            try:
                update_manifest_row(
                    self.manifest_path,
                    shot_id=body["shot_id"],
                    release_frame=body["release_frame"],
                    shooter_box=body["shooter_box"],
                    shooting_side=body["shooting_side"],
                    defender_box=body.get("defender_box"),
                )
                self.send_json({"ok": True})
            except Exception as e:
                self.send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def serve_file(self, filepath: Path):
        stat = filepath.stat()
        file_size = stat.st_size
        range_header = self.headers.get("Range")

        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                length = end - start + 1

                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with filepath.open("rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with filepath.open("rb") as f:
            self.wfile.write(f.read())


def start_annotation_server(
    manifest_path: str | Path,
    video_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    weights: str = "yolo11n-pose.pt",
    device: str = "cpu",
    open_browser: bool = True,
) -> None:
    """Launch the interactive web annotation server."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    v_dir = Path(video_dir) if video_dir else None

    handler = AnnotationRequestHandler
    handler.manifest_path = manifest_path
    handler.video_dir = v_dir
    handler.weights = weights
    handler.device = device

    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print("\n==================================================")
    print(f"🏀 NBA Shot Video Annotation Tool started at {url}")
    print(f"   Manifest: {manifest_path}")
    print("   Press Ctrl+C to stop the server.")
    print("==================================================\n")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server...")
    finally:
        server.server_close()
