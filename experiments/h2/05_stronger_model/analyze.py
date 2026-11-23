"""Score experiment 05 (GPT-6 Luna vs. qwen) as fixed in protocol.md.

Validation: raw Brier per model x condition, plus leave-one-game-out Platt.
Test: raw Brier, and Brier after a Platt map fitted on ALL of that model x
condition's validation predictions (no test labels are used for fitting).
Paired 95% intervals come from 1,000 game-bootstrap resamples (seed 42).

qwen ``multi`` / ``single_budget`` test predictions are taken from the frozen
02 run (same code and model); the anchored qwen test conditions come from this
experiment once that local run has finished.
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
metrics, platt, logo_platt, paired_bootstrap = (
    a04.metrics,
    a04.platt,
    a04.logo_platt,
    a04.paired_bootstrap,
)

MODELS = {
    "qwen": "ollama_{split}",
    "luna_none": "openai_none_{split}",
    "luna_low": "openai_low_{split}",
}
CONDITIONS = ["multi", "single_budget", "multi_anchored", "single_budget_anchored"]
FROZEN_02 = {"llm_multi": "qwen/multi", "llm_single_budget": "qwen/single_budget"}
BASELINES = ["xgboost", "logistic_regression", "constant"]


def load_run(root: Path, split: str) -> pd.DataFrame:
    frames = []
    for model, pattern in MODELS.items():
        path = root / pattern.format(split=split) / "predictions.jsonl"
        if not path.exists():
            continue
        records = [json.loads(x) for x in path.read_text().splitlines()]
        frame = pd.DataFrame(records)
        frame = frame[frame.status.eq("ok")].drop_duplicates(["condition", "shot_id"], keep="last")
        frame["column"] = model + "/" + frame["condition"]
        frames.append(frame[["shot_id", "column", "p_make"]])
    return pd.concat(frames)


def wide(selection: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    out = selection[["shot_id", "game_id", "y_true"]].copy()
    table = long.pivot_table(index="shot_id", columns="column", values="p_make")
    complete = [c for c in table.columns if table[c].notna().sum() == len(out)]
    return out.merge(table[complete], left_on="shot_id", right_index=True, how="left")


def compare(frame, pairs):
    rows = []
    for a, b in pairs:
        if a in frame and b in frame:
            delta, ci = paired_bootstrap(frame, a, b)
            rows.append({"a": a, "b": b, "delta_brier": delta, "ci95": ci})
    return rows


def pairs_for(columns, suffix=""):
    pairs = []
    for model in ["luna_none", "luna_low", "qwen"]:
        pairs += [
            (f"{model}/multi_anchored{suffix}", f"{model}/single_budget_anchored{suffix}"),
            (f"{model}/multi{suffix}", f"{model}/single_budget{suffix}"),
        ]
    for effort in ["luna_none", "luna_low"]:
        pairs += [(f"{effort}/{c}{suffix}", f"qwen/{c}{suffix}") for c in CONDITIONS]
    llm = [c for c in columns if "/" in c and c.endswith(suffix) and (suffix or "+" not in c)]
    return pairs, llm


def table(frame, columns):
    return {c: metrics(frame.y_true, frame[c]) for c in columns if c in frame}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/h2_stronger_model")
    parser.add_argument(
        "--val-baselines", default="artifacts/day1/models/validation_predictions.csv"
    )
    parser.add_argument(
        "--test-frozen", default="experiments/h2/02_final_test_100/report/predictions.csv"
    )
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # validation
    v_sel = pd.read_csv(root / "openai_none_validation" / "selection.csv")
    val = wide(v_sel, load_run(root, "validation"))
    base = pd.read_csv(args.val_baselines)
    for name in BASELINES:
        b = base[base.model.eq(name)][["shot_id", "p_make"]].rename(columns={"p_make": name})
        val = val.merge(b, on="shot_id", how="left")
    llm_val = [c for c in val.columns if "/" in c]
    for c in llm_val:
        val[c + "+platt_cv"] = logo_platt(
            val[["game_id", "y_true"]].assign(p_make=val[c]).reset_index(drop=True)
        )

    # test
    t_sel = pd.read_csv(root / "openai_none_test" / "selection.csv")
    long = load_run(root, "test")
    frozen = pd.read_csv(args.test_frozen)
    frozen = frozen[frozen.shot_id.isin(t_sel.shot_id)]
    for model, column in FROZEN_02.items():
        f = frozen[frozen.model.eq(model)][["shot_id", "p_make"]].assign(column=column)
        long = pd.concat([long[long.column.ne(column)], f])
    test = wide(t_sel, long)
    for name in BASELINES:
        b = frozen[frozen.model.eq(name)][["shot_id", "p_make"]].rename(columns={"p_make": name})
        test = test.merge(b, on="shot_id", how="left")
    llm_test = [c for c in test.columns if "/" in c]
    for c in llm_test:
        if c in val:
            test[c + "+recal"] = platt(val[c], val.y_true)(test[c])

    summary = {
        "validation": {
            "n": len(val),
            "make_rate": round(val.y_true.mean(), 3),
            "raw": table(val, BASELINES + llm_val),
            "platt_cv": table(val, [c + "+platt_cv" for c in llm_val]),
        },
        "test": {
            "n": len(test),
            "make_rate": round(test.y_true.mean(), 3),
            "raw": table(test, BASELINES + llm_test),
            "recalibrated": table(test, [c + "+recal" for c in llm_test]),
        },
    }

    raw_pairs, _ = pairs_for(test.columns)
    recal_pairs, _ = pairs_for(test.columns, "+recal")
    best = {
        split: min(
            (c for c in summary[split][key] if "/" in c),
            key=lambda c: summary[split][key][c]["brier"],
        )
        for split, key in [("validation", "raw"), ("test", "raw")]
    }
    best_recal = min(
        summary["test"]["recalibrated"], key=lambda c: summary["test"]["recalibrated"][c]["brier"]
    )
    summary["test"]["comparisons_raw"] = compare(test, raw_pairs + [(best["test"], "xgboost")])
    summary["test"]["comparisons_recal"] = compare(test, recal_pairs + [(best_recal, "xgboost")])
    summary["validation"]["comparisons_raw"] = compare(
        val, raw_pairs + [(best["validation"], "xgboost")]
    )
    summary["primary"] = next(
        (
            r
            for r in summary["test"]["comparisons_raw"]
            if r["a"] == "luna_none/multi_anchored" and r["b"] == "luna_none/single_budget_anchored"
        ),
        None,
    )

    val.to_csv(out / "validation_wide.csv", index=False)
    test.to_csv(out / "test_wide.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for split in ["validation", "test"]:
        print(f"\n## {split} (n={summary[split]['n']}, make rate {summary[split]['make_rate']})")
        for key in [k for k in summary[split] if not k.startswith(("n", "make", "comp"))]:
            print(f"\n{key}")
            for name, m in sorted(summary[split][key].items(), key=lambda kv: kv[1]["brier"]):
                print(
                    f"  {name:42s} brier {m['brier']:.4f}  ll {m['log_loss']:.4f}  "
                    f"auc {m['auc']:.3f}  mean_p {m['mean_p']:.3f}"
                )
        for key in [k for k in summary[split] if k.startswith("comp")]:
            print(f"\n{key}")
            for r in summary[split][key]:
                print(f"  {r['a']:40s} - {r['b']:40s} {r['delta_brier']:+.4f} {r['ci95']}")
    print("\nPRIMARY:", summary["primary"])


if __name__ == "__main__":
    main()
