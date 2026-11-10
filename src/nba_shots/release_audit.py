"""A frozen, blinded visual audit of the unchanged BasketEvent detector.

The reviewer reads only blind/ until references have been frozen. Automatic
predictions and source tracks stay in private/ until the explicit unblind stage.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import random
import ssl
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from .common import sha256, utc_now, write_json

DETECTOR_COMMIT = "8a45ea090e5b1bfaae50ceca1460076a5e81305c"
DETECTOR_SHA256 = "d2b1cd1e047894f73095b6037a21b014856cafd3c24d35a49231d19a261354b0"
BLIND_COLUMNS = ["shot_id", "clip_path"]
REFERENCE_STATUSES = {"clear", "uncertain", "not_visible", "unreadable", "unaligned"}
ROLE_STATUSES = {"clear", "uncertain", "not_visible", "none_visible"}


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_sample(rows, n=50, seed=42):
    ids = [r["shot_id"] for r in rows]
    if any(not str(s).strip() for s in ids) or len(ids) != len(set(ids)):
        raise ValueError("Pool must have unique, nonempty shot IDs")
    if not 1 <= n <= len(rows):
        raise ValueError("Sample size must be between one and the pool size")
    ordered = sorted(rows, key=lambda r: r["shot_id"])
    return random.Random(seed).sample(ordered, n)


def video_metadata(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frame_count": int(stream["nb_read_frames"]),
    }


def _fetch(url, destination):
    import certifi

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    for attempt in range(4):
        try:
            with urlopen(
                url, timeout=90, context=ssl.create_default_context(cafile=certifi.where())
            ) as response:
                content = response.read()
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def automatic_record(row):
    """Record original selected entity keys as well as the unchanged detector output."""
    import numpy as np

    from .basketevent import _at, _feet, boxes, clip_features

    path = Path(row["label_path"])
    clip = json.loads(path.read_text())
    keys = json.loads(row["shooter_keys"])
    result = {
        "shot_id": row["shot_id"],
        "label_path": str(path.resolve()),
        "label_sha256": sha256(path),
        "shooter_key": None,
        "defender_key": None,
        "release_frame": None,
        "track_frame_count": None,
    }
    if len(keys) != 1 or row.get("multiple_events") == "True":
        return {**result, "automatic_status": "ambiguous_shooter_or_event"}
    key = keys[0]
    result["shooter_key"] = key
    result["track_frame_count"] = len(boxes(clip[key]))
    try:
        features = clip_features(clip, key)
        r = features["release_frame"]
        result.update(release_frame=r, automatic_status=features["box_status"])
        if r is not None:
            shooter = boxes(clip[key])
            team = clip[key].get("jersey_color")
            candidates = []
            for other_key, entity in clip.items():
                if other_key in (key, "ball") or not isinstance(entity, dict):
                    continue
                if team is not None and entity.get("jersey_color") not in (None, team):
                    distance = np.linalg.norm(_feet(_at(boxes(entity), r)) - _feet(shooter[r]))
                    if np.isfinite(distance):
                        candidates.append((other_key, float(distance)))
            if candidates:
                result["defender_key"] = min(candidates, key=lambda p: p[1])[0]
    except Exception as error:
        result["automatic_status"] = "error"
        result["automatic_error"] = str(error)
    return result


def sample_audit(pool, output, n=50, seed=42, expected_population=851, download=False):
    if sha256(Path(__file__).with_name("basketevent.py")) != DETECTOR_SHA256:
        raise ValueError("Detector differs from the frozen 8a45ea0 implementation")
    rows, output = read_csv(pool), Path(output)
    if len(rows) != expected_population:
        raise ValueError(f"Verified pool has {len(rows)} rows, expected {expected_population}")
    selected = select_sample(rows, n, seed)
    if output.exists():
        raise ValueError("Audit output must be new; preserve frozen selections")
    (output / "private").mkdir(parents=True)
    (output / "blind").mkdir()
    write_csv(output / "private" / "pool.csv", rows, list(rows[0]))
    write_csv(output / "private" / "selection.csv", selected, list(rows[0]))
    if download:

        def fetch_row(row):
            try:
                _fetch(row["video_url"], Path(row["clip_path"]))
            except Exception:
                pass  # Keep acquisition failures in their selected rows below.

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(fetch_row, selected))
    automatic, blind, references = [], [], []
    for row in selected:
        record = automatic_record(row)
        video = Path(row["clip_path"])
        record["clip_sha256"] = None
        try:
            if download:
                _fetch(row["video_url"], video)
            record["clip_sha256"] = sha256(video)
            metadata = video_metadata(video)
            record["video_metadata"] = metadata
            record["alignment_status"] = (
                "count_equal_unverified"
                if metadata["frame_count"] == record["track_frame_count"]
                else "frame_count_mismatch"
            )
        except Exception as error:
            record.update(acquisition_error=str(error), alignment_status="unavailable")
        automatic.append(record)
        blind.append({"shot_id": row["shot_id"], "clip_path": str(video.resolve())})
        references.append(
            {
                "shot_id": row["shot_id"],
                "status": "",
                "release_frame": None,
                "alignment_verified": False,
                "shooter_status": "",
                "shooter_box": None,
                "defender_status": "",
                "defender_box": None,
                "evidence": [],
                "notes": "",
            }
        )
    write_csv(output / "blind" / "manifest.csv", blind, BLIND_COLUMNS)
    write_json(output / "blind" / "references.json", references)
    write_json(output / "private" / "automatic.json", automatic)
    write_json(
        output / "freeze.json",
        {
            "created_at": utc_now(),
            "detector_commit": DETECTOR_COMMIT,
            "seed": seed,
            "population": len(rows),
            "sample_size": n,
            "pool_sha256": sha256(pool),
            "selection_sha256": sha256(output / "private" / "selection.csv"),
            "automatic_sha256": sha256(output / "private" / "automatic.json"),
            "blind_manifest_sha256": sha256(output / "blind" / "manifest.csv"),
            "detector_source_sha256": sha256(Path(__file__).with_name("basketevent.py")),
            "reviewer": "assistant visual review",
            "definition": "first frame with ball free of hand",
            "defender_definition": (
                "nearest opposing player by image-space feet distance at release"
            ),
        },
    )
    return {
        "audit": str(output),
        "population": len(rows),
        "sample_size": n,
        "review_manifest": str(output / "blind" / "manifest.csv"),
    }


def verify_freeze(audit):
    audit = Path(audit)
    freeze = json.loads((audit / "freeze.json").read_text())
    for file, key in [
        ("private/automatic.json", "automatic_sha256"),
        ("private/selection.csv", "selection_sha256"),
        ("blind/manifest.csv", "blind_manifest_sha256"),
    ]:
        if sha256(audit / file) != freeze[key]:
            raise ValueError(f"Frozen input changed: {file}")
    return freeze


def freeze_references(audit):
    audit = Path(audit)
    freeze = verify_freeze(audit)
    marker = audit / "reference_freeze.json"
    if marker.exists():
        raise ValueError("References are already frozen")
    refs = json.loads((audit / "blind" / "references.json").read_text())
    selected = read_csv(audit / "blind" / "manifest.csv")
    selected_by_id = {r["shot_id"]: r for r in selected}
    auto_by_id = {
        r["shot_id"]: r for r in json.loads((audit / "private" / "automatic.json").read_text())
    }
    if len(refs) != freeze["sample_size"] or {r["shot_id"] for r in refs} != {
        r["shot_id"] for r in selected
    }:
        raise ValueError("Every sampled shot must have exactly one reference")
    if len({r["shot_id"] for r in refs}) != len(refs):
        raise ValueError("Duplicate reference shot IDs")
    for ref in refs:
        if ref["status"] not in REFERENCE_STATUSES:
            raise ValueError("Every reference needs a completed review status")
        if ref["status"] == "clear":
            r = ref["release_frame"]
            if isinstance(r, bool) or not isinstance(r, int) or r < 0:
                raise ValueError("Clear references need a zero-based integer release frame")
            if not ref["alignment_verified"] or not ref["evidence"]:
                raise ValueError("Clear references need verified alignment and saved evidence")
            metadata = auto_by_id[ref["shot_id"]].get("video_metadata", {})
            if r >= metadata.get("frame_count", 0):
                raise ValueError("Reference frame is outside the frozen decoded clip")
        clip_hash = auto_by_id[ref["shot_id"]].get("clip_sha256")
        if clip_hash and sha256(selected_by_id[ref["shot_id"]]["clip_path"]) != clip_hash:
            raise ValueError("Clip changed during reference review")
        if any(not Path(e).is_file() for e in ref["evidence"]):
            raise ValueError("Reference evidence is missing")
        for role in ["shooter", "defender"]:
            if ref[f"{role}_status"] not in ROLE_STATUSES:
                raise ValueError("Both player roles need review statuses")
            if ref[f"{role}_status"] == "clear":
                box = ref[f"{role}_box"]
                if (
                    not isinstance(box, list)
                    or len(box) != 4
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                    or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])
                ):
                    raise ValueError("Clear actor references need finite image-space boxes")
        identity_frame = ref.get("identity_frame")
        if identity_frame is not None:
            count = auto_by_id[ref["shot_id"]].get("video_metadata", {}).get("frame_count", 0)
            if (
                isinstance(identity_frame, bool)
                or not isinstance(identity_frame, int)
                or not 0 <= identity_frame < count
            ):
                raise ValueError("Identity frame is outside the frozen decoded clip")
    write_json(
        marker,
        {
            "created_at": utc_now(),
            "references_sha256": sha256(audit / "blind" / "references.json"),
            "automatic_sha256": freeze["automatic_sha256"],
        },
    )
    return {"references_frozen": len(refs)}


def checked_references(audit):
    verify_freeze(audit)
    marker = json.loads((Path(audit) / "reference_freeze.json").read_text())
    if sha256(Path(audit) / "blind" / "references.json") != marker["references_sha256"]:
        raise ValueError("Reference annotations changed after freezing")
    return json.loads((Path(audit) / "blind" / "references.json").read_text())


def render_frames(audit, shot_id, start=0, stop=None, step=25, width=384):
    """Uniform/indexed source frames only; never reads private automatic predictions."""
    from PIL import Image, ImageDraw

    audit = Path(audit)
    row = next(r for r in read_csv(audit / "blind" / "manifest.csv") if r["shot_id"] == shot_id)
    metadata = video_metadata(row["clip_path"])
    stop = metadata["frame_count"] - 1 if stop is None else stop
    if not (0 <= start <= stop < metadata["frame_count"]) or step < 1:
        raise ValueError("Invalid decoded-frame range")
    folder = audit / "blind" / "frames" / shot_id.replace(":", "_")
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="release_frames_") as temporary:
        pattern = str(Path(temporary) / "%05d.png")
        select = f"select=between(n\\,{start}\\,{stop})*not(mod(n-{start}\\,{step}))"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                row["clip_path"],
                "-vf",
                select,
                "-fps_mode",
                "vfr",
                pattern,
            ],
            check=True,
        )
        paths = sorted(Path(temporary).glob("*.png"))
        tiles, indices = [], list(range(start, stop + 1, step))
        if len(paths) != len(indices):
            raise ValueError("Decoded frame count disagrees with the requested indices")
        for path, index in zip(paths, indices):
            original = Image.open(path).convert("RGB")
            original.save(folder / f"frame_{index:05d}.png")
            height = round(original.height * width / original.width)
            tile = Image.new("RGB", (width, height + 26), "white")
            tile.paste(original.resize((width, height)), (0, 26))
            ImageDraw.Draw(tile).text((5, 5), f"{shot_id} | frame {index}", fill="black")
            tiles.append(tile)
        columns = min(4, len(tiles))
        sheet = Image.new(
            "RGB", (columns * width, math.ceil(len(tiles) / columns) * tiles[0].height), "#dddddd"
        )
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % columns) * width, (i // columns) * tile.height))
        output = folder / f"sheet_{start}_{stop}_{step}.jpg"
        sheet.save(output, quality=94)
    return {"sheet": str(output), **metadata}


def unblind_audit(audit):
    import numpy as np
    from PIL import Image, ImageDraw

    from .basketevent import _at, boxes

    audit = Path(audit)
    refs = checked_references(audit)
    automatic = json.loads((audit / "private" / "automatic.json").read_text())
    # Match by ID even if annotations were reordered during review.
    auto_by_id = {a["shot_id"]: a for a in automatic}
    packet = [{"reference": r, "automatic": auto_by_id[r["shot_id"]]} for r in refs]
    destination = audit / "unblinded"
    destination.mkdir(exist_ok=True)
    write_json(destination / "comparison_packet.json", packet)
    for ref in refs:
        frame_index = ref["release_frame"]
        if frame_index is None:
            frame_index = ref.get("identity_frame")
        if frame_index is None:
            continue
        shot = ref["shot_id"].replace(":", "_")
        frame_path = audit / "blind" / "frames" / shot / f"frame_{frame_index:05d}.png"
        if not frame_path.exists():
            render_frames(audit, ref["shot_id"], frame_index, frame_index, 1)
        image = Image.open(frame_path).convert("RGB")
        drawing = ImageDraw.Draw(image)
        auto = auto_by_id[ref["shot_id"]]
        if sha256(auto["label_path"]) != auto["label_sha256"]:
            raise ValueError("Frozen source tracking labels changed")
        clip = json.loads(Path(auto["label_path"]).read_text())
        for role, colour in [("shooter", "red"), ("defender", "yellow")]:
            key = auto[f"{role}_key"]
            if key is not None:
                box = _at(boxes(clip[key]), frame_index)
                if np.isfinite(box).all():
                    x, y, w, h = box.tolist()
                    drawing.rectangle([x, y, x + w, y + h], outline=colour, width=4)
                    drawing.text((x, max(0, y - 16)), f"automatic {role}: {key}", fill=colour)
            reference_box = ref[f"{role}_box"]
            if reference_box:
                drawing.rectangle(reference_box, outline="cyan", width=2)
        drawing.text((10, 10), f"{ref['shot_id']} | reference frame {frame_index}", fill="cyan")
        image.save(destination / f"{shot}_identity.png")
    identities = [
        {
            "shot_id": r["shot_id"],
            "shooter_match": None,
            "defender_match": None,
            "notes": "",
            "evidence": [],
        }
        for r in refs
    ]
    if not (destination / "identity_reviews.json").exists():
        write_json(destination / "identity_reviews.json", identities)
    return {"packet": str(destination / "comparison_packet.json")}


def wilson_interval(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, denominator = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def compare_rows(references, automatic, identities):
    ids = [r["shot_id"] for r in references]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reference IDs")
    for rows in [automatic, identities]:
        if len(rows) != len(ids) or {r["shot_id"] for r in rows} != set(ids):
            raise ValueError("Comparison must cover every sampled shot exactly once")
    autos, roles = ({r["shot_id"]: r for r in rows} for rows in [automatic, identities])
    comparisons = []
    for ref in references:
        auto, identity = autos[ref["shot_id"]], roles[ref["shot_id"]]
        scorable = ref["status"] == "clear" and ref["alignment_verified"]
        predicted = auto["release_frame"]
        error = predicted - ref["release_frame"] if scorable and predicted is not None else None
        category = (
            "unscorable"
            if not scorable
            else "missing"
            if error is None
            else "within_3"
            if abs(error) <= 3
            else "early"
            if error < -3
            else "late"
        )
        result = {
            "shot_id": ref["shot_id"],
            "reference_status": ref["status"],
            "reference_frame": ref["release_frame"],
            "automatic_frame": predicted,
            "scorable": scorable,
            "signed_error": error,
            "timing_category": category,
        }
        for role in ["shooter", "defender"]:
            eligible = ref["alignment_verified"] and ref[f"{role}_status"] in {
                "clear",
                "none_visible",
            }
            value = identity[f"{role}_match"]
            if eligible:
                if auto.get(f"{role}_key") is None:
                    value = ref[f"{role}_status"] == "none_visible"
                elif not isinstance(value, bool):
                    raise ValueError("Scorable identities require visual agreement judgments")
            result[f"{role}_scorable"] = eligible
            result[f"{role}_match"] = value if eligible else None
        comparisons.append(result)
    return comparisons


def summarize(comparisons):
    n = sum(r["scorable"] for r in comparisons)
    k = sum(r["timing_category"] == "within_3" for r in comparisons)
    errors = sorted(abs(r["signed_error"]) for r in comparisons if r["signed_error"] is not None)
    median = (
        None
        if not errors
        else (
            errors[len(errors) // 2]
            if len(errors) % 2
            else sum(errors[len(errors) // 2 - 1 : len(errors) // 2 + 1]) / 2
        )
    )
    result = {
        "sample_size": len(comparisons),
        "reference_scorable": n,
        "within_3": k,
        "accuracy": k / n if n else None,
        "wilson_95": wilson_interval(k, n),
        "reference_coverage": n / len(comparisons),
        "confirmed_matches_out_of_sample": k / len(comparisons),
        "median_absolute_frame_error": median,
        "timing_counts": {
            s: sum(r["timing_category"] == s for r in comparisons)
            for s in ["within_3", "early", "late", "missing", "unscorable"]
        },
    }
    for role in ["shooter", "defender"]:
        total = sum(r[f"{role}_scorable"] for r in comparisons)
        correct = sum(r[f"{role}_match"] is True for r in comparisons)
        result[role] = {
            "correct": correct,
            "scorable": total,
            "accuracy": correct / total if total else None,
            "wilson_95": wilson_interval(correct, total),
        }
    return result


def score_audit(audit, output):
    audit, output = Path(audit), Path(output)
    refs = checked_references(audit)
    automatic = json.loads((audit / "private" / "automatic.json").read_text())
    identities = json.loads((audit / "unblinded" / "identity_reviews.json").read_text())
    rows = compare_rows(refs, automatic, identities)
    metrics = summarize(rows)
    if output.exists():
        raise ValueError("Score output must be new")
    output.mkdir(parents=True)
    write_csv(output / "comparisons.csv", rows, list(rows[0]))
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "scoring_provenance.json",
        {
            "created_at": utc_now(),
            "reviewer": "assistant visual review",
            "audit_freeze_sha256": sha256(audit / "freeze.json"),
            "reference_freeze_sha256": sha256(audit / "reference_freeze.json"),
            "references_sha256": sha256(audit / "blind/references.json"),
            "identity_reviews_sha256": sha256(audit / "unblinded/identity_reviews.json"),
            "scorer_sha256": sha256(Path(__file__)),
            "reference_status_counts": {
                status: sum(r["status"] == status for r in refs)
                for status in sorted(REFERENCE_STATUSES)
            },
        },
    )
    k, n, total = metrics["within_3"], metrics["reference_scorable"], metrics["sample_size"]
    interval = metrics["wilson_95"]
    wording = (
        f"Automatic release detection agreed within ±3 frames with assistant-reviewed "
        f"references on **{k}/{n} shots ({100 * k / n:.1f}%; 95% Wilson CI "
        f"{100 * interval[0]:.1f}–{100 * interval[1]:.1f}%)**."
        if n
        else "No sampled clips had a scorable release reference; accuracy is undefined."
    )
    lines = [
        "# Blind release detection audit",
        "",
        wording,
        f"{n}/{total} sampled clips were scorable. Confirmed timing matches across all "
        f"sampled clips: {k}/{total}.",
        "",
        "References are assistant visual annotations, "
        "not independent human ground truth. Conclusions apply only to the frozen "
        "video-available subset. Missing automatic detections count as failures when a "
        "clear aligned reference exists. Unscorable references are retained, not replaced.",
        "",
        f"Timing categories: {json.dumps(metrics['timing_counts'])}.",
        f"Median absolute error among detected, scorable shots: "
        f"{metrics['median_absolute_frame_error']} frames.",
        "",
    ]
    for role in ["shooter", "defender"]:
        m = metrics[role]
        lines.append(f"- {role.title()} identity agreement: {m['correct']}/{m['scorable']}.")
    lines += ["", "## Cases for review", ""]
    for row in rows:
        if (
            row["timing_category"] != "within_3"
            or row["shooter_match"] is False
            or row["defender_match"] is False
        ):
            lines.append(
                f"- {row['shot_id']}: timing={row['timing_category']}; "
                f"error={row['signed_error']}; shooter={row['shooter_match']}; "
                f"defender={row['defender_match']}."
            )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return {"report": str(output / "report.md"), **metrics}
