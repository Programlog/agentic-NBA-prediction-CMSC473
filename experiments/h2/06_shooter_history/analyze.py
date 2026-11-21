"""Score experiment 06 (shooter history) against 05's no-history runs and the baselines.

Columns on the same 100 shots per split:
  hist/{multi,single}, named/{multi,single}   this experiment (GPT-6 Luna, effort none)
  nohist/{multi,single}                       05 Luna none, anchored (same prompts minus history)
  xgboost, logistic_regression, constant      frozen day-1 baselines
  xgboost_hist, logistic_regression_hist      retrained with the same history features
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "analyze04", HERE.parent / "04_brier_followup" / "analyze.py"
)
a04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a04)

RENAME = {
    "multi_hist": "hist/multi",
    "single_hist": "hist/single",
    "multi_named": "named/multi",
    "single_named": "named/single",
}
FROM_05 = {
    "luna_none/multi_anchored": "nohist/multi",
    "luna_none/single_budget_anchored": "nohist/single",
}
BASE = ["xgboost", "logistic_regression", "constant"]
PRIMARY = [("hist/multi", "nohist/multi"), ("named/multi", "hist/multi")]
SECONDARY = [
    ("hist/single", "nohist/single"),
    ("named/single", "hist/single"),
    ("hist/multi", "hist/single"),
    ("named/multi", "named/single"),
    ("hist/multi", "xgboost_hist"),
    ("named/multi", "xgboost_hist"),
    ("named/single", "xgboost_hist"),
    ("xgboost_hist", "xgboost"),
    ("logistic_regression_hist", "logistic_regression"),
]


def build(split: str, root: Path, wide05: Path, baselines: Path) -> pd.DataFrame:
    run = root / f"openai_none_{split}"
    records = [json.loads(x) for x in (run / "predictions.jsonl").read_text().splitlines()]
    preds = pd.DataFrame(records)
    preds = preds[preds.status.eq("ok")].drop_duplicates(["condition", "shot_id"], keep="last")
    frame = pd.read_csv(wide05)[["shot_id", "game_id", "y_true", *BASE, *FROM_05]].rename(
        columns=FROM_05
    )
    table = preds.pivot(index="shot_id", columns="condition", values="p_make").rename(
        columns=RENAME
    )
    frame = frame.merge(table, left_on="shot_id", right_index=True, how="left")
    base = pd.read_csv(baselines)
    base = base[base.shot_id.isin(frame.shot_id)].pivot(
        index="shot_id", columns="model", values="p_make"
    )
    return frame.merge(base, left_on="shot_id", right_index=True, how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/h2_history/runs")
    parser.add_argument("--baselines", default="artifacts/h2_history/baseline_predictions.csv")
    parser.add_argument("--results05", default=str(HERE.parent / "05_stronger_model" / "results"))
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits:
        frame = build(
            split,
            Path(args.root),
            Path(args.results05) / f"{split}_wide.csv",
            Path(args.baselines),
        )
        cols = [c for c in frame.columns if c not in {"shot_id", "game_id", "y_true"}]
        missing = [c for c in cols if frame[c].isna().any()]
        if missing:
            raise SystemExit(f"{split}: incomplete columns {missing}")
        frame.to_csv(out / f"{split}_wide.csv", index=False)
        comps = []
        for role, pairs in [("primary", PRIMARY), ("secondary", SECONDARY)]:
            for a, b in pairs:
                delta, ci = a04.paired_bootstrap(frame, a, b)
                comps.append({"role": role, "a": a, "b": b, "delta_brier": delta, "ci95": ci})
        summary[split] = {
            "n": len(frame),
            "make_rate": round(float(frame.y_true.mean()), 3),
            "metrics": {c: a04.metrics(frame.y_true, frame[c]) for c in cols},
            "comparisons": comps,
        }
        print(f"\n## {split} (n={len(frame)}, make rate {frame.y_true.mean():.2f})")
        for name, m in sorted(summary[split]["metrics"].items(), key=lambda kv: kv[1]["brier"]):
            print(
                f"  {name:26s} brier {m['brier']:.4f}  auc {m['auc']:.3f}  mean_p {m['mean_p']:.3f}"
            )
        for c in comps:
            print(
                f"  [{c['role'][:3]}] {c['a']:24s} - {c['b']:20s} "
                f"{c['delta_brier']:+.4f} {c['ci95']}"
            )
    contamination = Path(args.root).parent / "contamination.json"
    if contamination.exists():
        summary["contamination"] = json.loads(contamination.read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
