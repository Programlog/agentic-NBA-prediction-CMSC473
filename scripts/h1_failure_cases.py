"""H1 failure-case analysis, as fixed in docs/h1_failure_cases.md (exploratory, descriptive).

    python scripts/h1_failure_cases.py COHORT COLLECTION OUT

Part A reads the saved test-run predictions; part B the saved cross-validation predictions and
12 BasketEvent videos (downloaded to a temp folder and deleted). Nothing is refit or rescored.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nba_shots import basketevent as be
from nba_shots.common import BOX_FEATURES
from nba_shots.evaluation import paired_comparison

TEST = Path("reports/basketevent_h1_test/test_predictions_seed_mean.csv")
CV_RUNS = ["artifacts/h1_rebuild_cv_d0"] + [
    f"artifacts/h1_rebuild_cv_d0_seed{s}" for s in range(1, 6)
]
BOX, BASE = "xgboost_boxes", "xgboost"
REVIEW = 6


def wide(preds: pd.DataFrame) -> pd.DataFrame:
    table = preds.pivot_table(
        index=["shot_id", "game_id", "y_true"], columns="model", values="p_make"
    ).reset_index()
    return table.rename(columns={BOX: "p_box", BASE: "p_base"})[
        ["shot_id", "game_id", "y_true", "p_box", "p_base"]
    ]


def attach(table: pd.DataFrame, cohort: Path) -> pd.DataFrame:
    shots = pd.read_parquet(cohort / "shots.parquet")
    feats = pd.read_parquet(cohort / "box_features.parquet")
    cols = ["shot_id", "shot_distance", "shot_value", "action_type", "player_prior_make_rate"]
    out = table.merge(shots[cols], on="shot_id", how="left").merge(
        feats[["shot_id", "clip", "has_video", "release_frame", *sorted(BOX_FEATURES)]],
        on="shot_id",
        how="left",
    )
    out["low_release"] = out.release_height_bh < 0.9
    out["box_missing"] = out[sorted(BOX_FEATURES)].isna().any(axis=1)
    out["fifth"] = pd.qcut(out.p_box, 5, labels=False) + 1
    return out


def profile(rows: pd.DataFrame) -> dict:
    return {
        "shots": len(rows),
        "p_box_mean": float(rows.p_box.mean()),
        "shot_distance_ft": float(rows.shot_distance.mean()),
        "three_share": float(rows.shot_value.eq(3).mean()),
        "release_height_bh": float(rows.release_height_bh.mean()),
        "low_release_share": float(rows.low_release.mean()),
        "defender_distance_bh": float(rows.defender_distance_bh.mean()),
        "box_missing_share": float(rows.box_missing.mean()),
        "prior_make_rate": float(rows.player_prior_make_rate.mean()),
        "top_shot_types": rows.action_type.value_counts(normalize=True).head(3).round(3).to_dict(),
    }


def slice_name(row) -> dict:
    d = row.shot_distance
    action = str(row.action_type).lower()
    return {
        "2 vs 3": "3-pointer" if row.shot_value == 3 else "2-pointer",
        "distance": "3-pointer"
        if row.shot_value == 3
        else ("under 10 ft" if d < 10 else "10-16 ft" if d < 16 else "16 ft to the line"),
        "release height": "below 0.9 bh" if row.low_release else "0.9 bh or higher",
        "box features": "some missing" if row.box_missing else "all present",
        "shot type": "Pullup"
        if "pullup" in action
        else "Step Back"
        if "step back" in action
        else "Jump Shot"
        if action == "jump shot"
        else "other",
    }


def part_a(cohort: Path) -> dict:
    t = attach(wide(pd.read_csv(TEST, dtype={"game_id": str})), cohort)
    deciles = pd.qcut(t.p_box, 10, labels=False) + 1
    calibration = [
        {
            "decile": int(k),
            "shots": len(g),
            "p_box": float(g.p_box.mean()),
            "p_base": float(g.p_base.mean()),
            "made": float(g.y_true.mean()),
        }
        for k, g in t.groupby(deciles)
    ]
    low, high = t.fifth.eq(1), t.fifth.eq(5)
    groups = {
        "surprising_makes": t[low & t.y_true.eq(1)],
        "expected_misses_same_fifth": t[low & t.y_true.eq(0)],
        "surprising_misses": t[high & t.y_true.eq(0)],
        "expected_makes_same_fifth": t[high & t.y_true.eq(1)],
        "all_test_shots": t,
    }
    change = (t.p_box - t.p_base).abs()
    moved = t[change >= change.quantile(0.9)]
    toward = np.sign(moved.p_box - moved.p_base) == np.where(moved.y_true == 1, 1, -1)
    err = lambda p, y: ((p - y) ** 2).mean()  # noqa: E731
    changed = {
        **profile(moved),
        "mean_abs_change": float(change[moved.index].mean()),
        "share_moved_toward_outcome": float(toward.mean()),
        "brier_box_minus_base": float(
            err(moved.p_box, moved.y_true) - err(moved.p_base, moved.y_true)
        ),
        "moved_down_share": float((moved.p_box < moved.p_base).mean()),
    }
    names = pd.DataFrame([slice_name(r) for r in t.itertuples()], index=t.index)
    long = []
    for model, col in [(BASE, "p_base"), (BOX, "p_box")]:
        long.append(
            t.assign(model=model, p_make=t[col], cohort=be.COHORT, split="test")[
                ["shot_id", "game_id", "cohort", "split", "y_true", "model", "p_make"]
            ]
        )
    long = pd.concat(long)
    slices = []
    for dim in names:
        for level, idx in names.groupby(dim).groups.items():
            ids = set(t.loc[idx, "shot_id"])
            part = long[long.shot_id.isin(ids)]
            res = paired_comparison(part, BASE, BOX)
            slices.append(
                {
                    "slice": dim,
                    "level": level,
                    "shots": len(ids),
                    "games": int(t.loc[idx, "game_id"].nunique()),
                    "made": float(t.loc[idx, "y_true"].mean()),
                    "brier_box_minus_base": res["brier_delta_candidate_minus_baseline"],
                    "ci95": res["brier_delta_95pct_interval"],
                }
            )
    return {
        "shots": len(t),
        "calibration": calibration,
        "profiles": {k: profile(v) for k, v in groups.items()},
        "changed_by_boxes": changed,
        "slices": slices,
    }


def part_b(cohort: Path, collection: Path, out: Path) -> list[dict]:
    import cv2
    from huggingface_hub import hf_hub_download

    frames = [pd.read_csv(Path(r) / "cv_predictions.csv", dtype={"game_id": str}) for r in CV_RUNS]
    cv = pd.concat(frames)
    cv = (
        cv[cv.model.isin([BOX, BASE])]
        .groupby(["shot_id", "game_id", "y_true", "model"], as_index=False)
        .p_make.mean()
    )
    t = attach(wide(cv), cohort)
    video = t[t.has_video.astype(bool)]
    picks = pd.concat(
        [
            video[video.y_true.eq(1)].nsmallest(REVIEW, "p_box").assign(kind="surprising make"),
            video[video.y_true.eq(0)].nlargest(REVIEW, "p_box").assign(kind="surprising miss"),
        ]
    )
    revision = json.loads((collection / "source.json").read_text())["revision"]
    review_dir = out / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(picks.itertuples(), 1):
        game = row.clip.split("/")[1]
        with gzip.open(collection / "games" / f"{game}_labels.json.gz", "rt") as stream:
            clip = json.load(stream)[row.clip]
        key = next(
            k
            for k, v in clip.items()
            if k != "ball"
            and isinstance(v, dict)
            and isinstance(v.get("event"), dict)
            and v["event"].get("actionType") in be.SHOT_ACTIONS
        )
        shooter = be.steady(be.boxes(clip[key]))
        ball = be.boxes(clip.get("ball") or {})
        team = clip[key].get("jersey_color")
        r = int(row.release_frame)
        opponents = [
            be.steady(be.boxes(v))
            for k, v in clip.items()
            if k not in (key, "ball")
            and isinstance(v, dict)
            and team is not None
            and v.get("jersey_color") not in (None, team)
        ]
        h = np.nanmedian(be._stance(shooter, r)[:, 3])
        dists = [be._court_distance(be._at(o, r), shooter[r], h) for o in opponents]
        defender = (
            opponents[int(np.nanargmin(dists))] if dists and np.isfinite(dists).any() else None
        )
        tmp = Path(tempfile.mkdtemp(prefix="h1fail_"))
        try:
            mp4 = Path(
                hf_hub_download(
                    be.REPO,
                    row.clip.replace(".json", ".mp4"),
                    repo_type="dataset",
                    revision=revision,
                    local_dir=tmp,
                )
            )
            cap = cv2.VideoCapture(str(mp4))
            panels = []
            for f in (r - 10, r, r + 4):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(f, 0))
                ok, img = cap.read()
                if not ok:
                    img = np.zeros((720, 1280, 3), np.uint8)
                for track, colour in [
                    (shooter, (214, 120, 42)),
                    (defender, (52, 104, 235)),
                    (ball, (0, 161, 237)),
                ]:
                    if track is None:
                        continue
                    b = be._at(track, f)
                    if np.isfinite(b).all():
                        x, y, w, hh = b.astype(int)
                        cv2.rectangle(img, (x, y), (x + w, y + hh), colour, 3)
                label = {r - 10: "r-10", r: "release r", r + 4: "r+4"}[f]
                cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
                panels.append(cv2.resize(img, (640, 360)))
            cap.release()
            name = f"{i:02d}_{row.kind.replace(' ', '_')}.jpg"
            cv2.imwrite(str(review_dir / name), np.hstack(panels))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rows.append(
            {
                "image": f"review/{name}",
                "kind": row.kind,
                "shot_id": row.shot_id,
                "made": int(row.y_true),
                "p_box": round(float(row.p_box), 3),
                "p_base": round(float(row.p_base), 3),
                "action_type": row.action_type,
                "shot_distance_ft": row.shot_distance,
                "release_height_bh": round(float(row.release_height_bh), 2),
                "defender_distance_bh": round(float(row.defender_distance_bh), 2),
            }
        )
    pd.DataFrame(rows).to_csv(out / "review.csv", index=False)
    return rows


def main(cohort: Path, collection: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "docs/h1_failure_cases.md", "test": part_a(cohort)}
    summary["video_review"] = part_b(cohort, collection, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
