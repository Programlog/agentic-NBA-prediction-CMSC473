"""Score experiment 07 against the frozen baselines.

Usage
  uv run --frozen --all-extras python experiments/h2/07_prompt_tuning/analyze.py \
      --winner V4 --nomodel V2 [--splits validation test]

Columns per split: every run condition plus xgboost, logistic_regression, knn50 and
constant from prepare.py. "fresh" = shots no earlier experiment used (test only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "analyze04", HERE.parent / "04_brier_followup" / "analyze.py"
)
a04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a04)

BASE = {
    "p_xgboost": "xgboost",
    "p_logistic_regression": "logistic_regression",
    "p_knn": "knn50",
    "p_constant": "constant",
}


def comparisons(winner: str, nomodel: str) -> list[tuple[str, str, str]]:
    w, m, s = winner, f"multi:{winner}", f"single_budget:{winner}"
    return [
        ("primary", m, "xgboost"),
        ("secondary", s, "xgboost"),
        ("secondary", m, s),
        ("secondary", w, "V0"),
        ("secondary", w, "xgboost"),
        ("secondary", nomodel, "xgboost"),
        ("secondary", nomodel, "V0"),
        ("secondary", "V0", "xgboost"),
        ("secondary", "knn50", "xgboost"),
    ]


def build(split: str, rows_dir: Path, runs: Path) -> pd.DataFrame:
    rows = pd.read_parquet(rows_dir / f"{split}_rows.parquet")
    frame = rows[["shot_id", "game_id", "y_true", "in_old_100", *BASE]].rename(columns=BASE)
    path = runs / f"openai_none_{split}" / "predictions.jsonl"
    preds = pd.DataFrame([json.loads(x) for x in path.read_text().splitlines()])
    preds = preds[preds.status.eq("ok")].drop_duplicates(["condition", "shot_id"], keep="last")
    table = preds.pivot(index="shot_id", columns="condition", values="p_make")
    return frame.merge(table, left_on="shot_id", right_index=True, how="left")


def score(frame: pd.DataFrame, cols: list[str], comps) -> dict:
    out = {
        "n": len(frame),
        "games": int(frame.game_id.nunique()),
        "make_rate": round(float(frame.y_true.mean()), 3),
        "metrics": {},
        "comparisons": [],
    }
    for c in cols:
        m = a04.metrics(frame.y_true, frame[c])
        m["copies_xgb"] = round(float((np.abs(frame[c] - frame.xgboost) < 0.005).mean()), 3)
        same = frame[c].std() == 0
        corr = None if same else round(float(np.corrcoef(frame[c], frame.xgboost)[0, 1]), 3)
        m["corr_xgb"] = corr
        out["metrics"][c] = m
    for role, a, b in comps:
        if a in cols and b in cols:
            delta, ci = a04.paired_bootstrap(frame, a, b)
            out["comparisons"].append(
                {"role": role, "a": a, "b": b, "delta_brier": delta, "ci95": ci}
            )
    return out


def report(name: str, s: dict):
    print(f"\n## {name} (n={s['n']}, {s['games']} games, make rate {s['make_rate']})")
    for c, m in sorted(s["metrics"].items(), key=lambda kv: kv[1]["brier"]):
        print(
            f"  {c:22s} brier {m['brier']:.4f}  auc {m['auc']:.3f}  mean_p {m['mean_p']:.3f}  "
            f"corr_xgb {m['corr_xgb'] or 0:.2f}  copies_xgb {m['copies_xgb']:.2f}"
        )
    for c in s["comparisons"]:
        print(
            f"  [{c['role'][:3]}] {c['a']:20s} - {c['b']:18s} {c['delta_brier']:+.4f} {c['ci95']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", default="artifacts/h2_prompts")
    parser.add_argument("--runs", default="artifacts/h2_prompts/runs")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--winner", default="V4")
    parser.add_argument("--nomodel", default="V2")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    comps = comparisons(args.winner, args.nomodel)
    summary = {"winner": args.winner, "nomodel": args.nomodel}
    for split in args.splits:
        frame = build(split, Path(args.rows_dir), Path(args.runs))
        cols = [c for c in frame.columns if c not in {"shot_id", "game_id", "y_true", "in_old_100"}]
        incomplete = [c for c in cols if frame[c].isna().any()]
        if incomplete:
            print(f"{split}: dropping incomplete columns {incomplete}")
            cols = [c for c in cols if c not in incomplete]
        frame.to_csv(out / f"{split}_wide.csv", index=False)
        summary[split] = score(frame, cols, comps)
        report(split, summary[split])
        if split == "test":
            fresh = frame[~frame.in_old_100]
            summary["test_fresh"] = score(fresh, cols, comps)
            report("test, fresh shots only", summary["test_fresh"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
