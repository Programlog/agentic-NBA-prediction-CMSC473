"""Contamination probe for arm B: how well does the model already know each shooter?

For every player in the validation and test selections, ask the model (no shot
context) for that player's 2014-15 regular-season field-goal percentage, and
compare with the FG% in this dataset. Very accurate recall would mean arm B may
be using memorized season outcomes rather than general player knowledge.

The FG% here is computed from the Kaggle shot logs (which cover most, not all, of
the season), so it is close to but not identical to official totals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nba_shots.agents import select_evaluation_rows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "05_stronger_model"))
from openai_client import PRICES, load_dotenv  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"fg_pct": {"type": "number"}, "confidence": {"type": "string"}},
    "required": ["fg_pct", "confidence"],
    "additionalProperties": False,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", default="artifacts/h2_history/shots.parquet")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--out", default="artifacts/h2_history/contamination.csv")
    args = parser.parse_args()
    load_dotenv()
    from openai import OpenAI

    api = OpenAI()
    shots = pd.read_parquet(args.shots)
    selected = pd.concat(
        [select_evaluation_rows(shots, s, 100, 42, 50) for s in ["validation", "test"]]
    )
    players = selected.drop_duplicates("player_id")[
        ["player_id", "player_name", "player_prior_make_rate"]
    ]
    season = shots.groupby("player_id")["y_true"].agg(data_fg="mean", data_attempts="size")
    out = Path(args.out)
    done = pd.read_csv(out, dtype={"player_id": str}) if out.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    seen = set(done["player_id"]) if len(done) else set()
    cost = 0.0
    for p in players.to_dict("records"):
        if p["player_id"] in seen:
            continue
        name = p["player_name"].title()
        r = api.chat.completions.create(
            model=args.model,
            reasoning_effort="none",
            temperature=0,
            seed=42,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"What was {name}'s field goal percentage in the 2014-15 NBA regular "
                        "season? Give your best estimate as a fraction between 0 and 1, and "
                        "say how confident you are (low/medium/high)."
                    ),
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "fg", "strict": True, "schema": SCHEMA},
            },
        )
        price = PRICES[args.model]
        cost += (
            r.usage.prompt_tokens * price["input"] + r.usage.completion_tokens * price["output"]
        ) / 1e6
        value = json.loads(r.choices[0].message.content)
        rows.append({**p, "llm_fg": value["fg_pct"], "llm_confidence": value["confidence"]})
        pd.DataFrame(rows).to_csv(out, index=False)
    frame = pd.DataFrame(rows).merge(season, left_on="player_id", right_index=True)
    frame.to_csv(out, index=False)
    llm_err = (frame["llm_fg"] - frame["data_fg"]).abs()
    prior_err = (frame["player_prior_make_rate"] - frame["data_fg"]).abs()
    summary = {
        "players": len(frame),
        "llm_corr": round(float(np.corrcoef(frame["llm_fg"], frame["data_fg"])[0, 1]), 3),
        "llm_mae": round(float(llm_err.mean()), 4),
        "llm_within_1pt": round(float((llm_err <= 0.01).mean()), 3),
        "prior_rate_corr": round(
            float(np.corrcoef(frame["player_prior_make_rate"], frame["data_fg"])[0, 1]), 3
        ),
        "prior_rate_mae": round(float(prior_err.mean()), 4),
        "confidence_counts": frame["llm_confidence"].str.lower().value_counts().to_dict(),
        "new_call_cost_usd": round(cost, 4),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
