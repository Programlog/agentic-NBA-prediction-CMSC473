"""H1 release-angle pilot, as fixed in docs/h1_release_angle_pilot.md.

    python scripts/h1_release_angle_pilot.py COHORT COLLECTION OUT

Box data only (no video, no outcomes, no test games). Writes OUT/angles.csv (one row per shot,
with the reason when not measured), OUT/summary.json and OUT/release_angle.png.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nba_shots import basketevent as be

WINDOW, FIT_FRAMES, MIN_FIT = 4, 12, 8
MIN_COS, MIN_FEET, MIN_OTHERS = 0.7, 8.0, 3
BODY_M = 2.0
SHOT_COLUMNS = ["shot_id", "game_id", "split", "shot_family", "blocked", "loc_x", "loc_y"]


def shooter_key(clip: dict) -> str:
    return next(
        k
        for k, v in clip.items()
        if k != "ball"
        and isinstance(v, dict)
        and isinstance(v.get("event"), dict)
        and v["event"].get("actionType") in be.SHOT_ACTIONS
    )


def camera_shift(others: list[np.ndarray], r: int, k: int) -> np.ndarray:
    """Median feet displacement of the other players from frame r to r+k (NaN if < 3 seen)."""
    moves = [be._feet(be._at(t, r + k)) - be._feet(be._at(t, r)) for t in others]
    moves = [m for m in moves if np.isfinite(m).all()]
    return np.median(moves, axis=0) if len(moves) >= MIN_OTHERS else np.full(2, np.nan)


def angle(pos: dict[int, np.ndarray], a: int, b: int, cos_phi: float) -> float:
    d = pos[b] - pos[a]
    if not np.isfinite(d).all():
        return np.nan
    return float(np.degrees(np.arctan2(-d[1], abs(d[0]) / cos_phi)))


def measure(clip: dict, r_stored: int, loc_x: float, loc_y: float) -> dict:
    key = shooter_key(clip)
    shooter = be.steady(be.boxes(clip[key]))
    ball = be.boxes(clip.get("ball") or {})
    r = be.release_frame(shooter, ball)
    out = {"release_frame_matches": r == r_stored}
    r = r_stored
    stance = be._stance(shooter, r)
    h = np.nanmedian(stance[:, 3]) if np.isfinite(stance).any() else np.nanmedian(shooter[:r, 3])
    others = [
        be.steady(be.boxes(v))
        for k, v in clip.items()
        if k not in (key, "ball") and isinstance(v, dict)
    ]
    # Camera-corrected ball centre, relative to the release frame's camera.
    pos = {}
    for k in range(FIT_FRAMES + 1):
        pos[k] = be._centre(be._at(ball, r + k)) - camera_shift(others, r, k)

    # Gravity check (validation only): quadratic fit over r .. r+12.
    ks = [k for k in pos if np.isfinite(pos[k]).all()]
    if len(ks) >= MIN_FIT and np.isfinite(h):
        a = np.polyfit(ks, [pos[k][1] for k in ks], 2)[0]
        out["g_ms2"] = float(2 * a * be.FPS**2 * BODY_M / h)

    distance = np.hypot(loc_x, loc_y) / 10  # NBA locations are in tenths of feet
    cos_phi = abs(loc_y) / np.hypot(loc_x, loc_y) if distance > 0 else 0.0
    out["cos_phi"] = float(cos_phi)
    if distance < MIN_FEET:
        return {**out, "status": "too_close"}
    if cos_phi < MIN_COS:
        return {**out, "status": "toward_or_away_from_camera"}
    if not np.isfinite(camera_shift(others, r, WINDOW)).all():
        return {**out, "status": "no_camera_reference"}
    if not np.isfinite(pos[WINDOW]).all():
        return {**out, "status": "ball_missing"}
    out["angle_deg"] = angle(pos, 0, WINDOW, cos_phi)
    out["angle_early_deg"] = angle(pos, 0, WINDOW // 2, cos_phi)
    out["angle_late_deg"] = angle(pos, WINDOW // 2, WINDOW, cos_phi)
    return {**out, "status": "ok"}


def spearman_ci(a, b, reps=1000, seed=42):
    from scipy.stats import spearmanr

    rho = float(spearmanr(a, b).statistic)
    rng = np.random.default_rng(seed)
    boot = [
        spearmanr(a[i], b[i]).statistic
        for i in (rng.integers(0, len(a), len(a)) for _ in range(reps))
    ]
    return rho, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def verdict(n: int, passed: bool) -> str:
    return "inconclusive (<100 shots)" if n < 100 else ("pass" if passed else "fail")


def main(cohort: Path, collection: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; choose a new output directory")
    shots = pd.read_parquet(cohort / "shots.parquet", columns=SHOT_COLUMNS)  # no y_true
    feats = pd.read_parquet(cohort / "box_features.parquet")
    rows = shots.merge(feats[["shot_id", "clip", "box_status", "release_frame"]], on="shot_id")
    rows = rows.loc[
        rows.split.isin(["train", "validation"])
        & rows.shot_family.eq("jump")
        & ~rows.blocked.astype(bool)
        & rows.box_status.eq("ok")
    ].sort_values("shot_id")
    records = []
    for game, part in rows.groupby(rows["clip"].str.split("/").str[1]):
        with gzip.open(collection / "games" / f"{game}_labels.json.gz", "rt") as stream:
            labels = json.load(stream)
        for row in part.itertuples():
            record = {"shot_id": row.shot_id, "game_id": row.game_id}
            try:
                record.update(
                    measure(labels[row.clip], int(row.release_frame), row.loc_x, row.loc_y)
                )
            except Exception as error:  # kept and counted
                record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
            records.append(record)
    table = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "angles.csv", index=False)

    ok = table.loc[table.status.eq("ok")]
    angles = ok.angle_deg.dropna()
    g = table.get("g_ms2", pd.Series(dtype=float)).dropna()
    halves = ok[["angle_early_deg", "angle_late_deg"]].dropna()
    rho, lo, hi = spearman_ci(halves.angle_early_deg.to_numpy(), halves.angle_late_deg.to_numpy())
    in_range = float(angles.between(20, 75).mean())
    summary = {
        "plan": "docs/h1_release_angle_pilot.md",
        "shots": len(table),
        "games": int(table.game_id.nunique()),
        "release_frame_matches": int(table.release_frame_matches.sum()),
        "status": table.status.value_counts().to_dict(),
        "gravity": {
            "measured": len(g),
            "median_ms2": float(g.median()),
            "iqr_ms2": [float(g.quantile(0.25)), float(g.quantile(0.75))],
            "verdict": verdict(len(g), 6.9 <= g.median() <= 12.7),
        },
        "plausible_range": {
            "measured": len(angles),
            "median_deg": float(angles.median()),
            "iqr_deg": [float(angles.quantile(0.25)), float(angles.quantile(0.75))],
            "share_20_to_75": in_range,
            "verdict": verdict(len(angles), 40 <= angles.median() <= 60 and in_range >= 0.8),
        },
        "reliability": {
            "measured": len(halves),
            "rho": rho,
            "ci": [lo, hi],
            "median_early_deg": float(halves.angle_early_deg.median()),
            "median_late_deg": float(halves.angle_late_deg.median()),
            "verdict": verdict(len(halves), rho >= 0.5 and lo > 0.3),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
    for ax, values, lim, band, title, unit in [
        (left, angles, (-30, 90), (40, 60), "Release angle from boxes", "degrees"),
        (right, g, (-10, 40), (6.9, 12.7), "Gravity from the ball's flight", "m/s²"),
    ]:
        ax.axvspan(*band, color="#cde2fb", zorder=0, label="pass range (median)")
        ax.hist(values.clip(*lim), bins=60, range=lim, color="#2a78d6", zorder=2)
        ax.axvline(values.median(), color="#0b0b0b", linewidth=1.5, zorder=3)
        ax.set_title(f"{title} (median {values.median():.1f})", loc="left", fontsize=10)
        ax.set_xlabel(unit, fontsize=9, color="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    left.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(
        0.01,
        0.01,
        "Development jump shots only; values outside the axis are piled at the edges.",
        fontsize=7.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "release_angle.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
