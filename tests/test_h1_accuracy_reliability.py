import numpy as np
import pandas as pd

from nba_shots.evaluation import score_predictions
from scripts.h1_accuracy_reliability import reliability


def test_reliability_uses_equal_count_bins_in_probability_order():
    rows = pd.DataFrame({"p_make": np.linspace(0.05, 0.95, 20), "y_true": [0] * 10 + [1] * 10})
    bins = reliability(rows, bins=4)
    assert [b["shots"] for b in bins] == [5, 5, 5, 5]
    assert [b["made"] for b in bins] == [0.0, 0.0, 1.0, 1.0]
    assert all(b["made_low"] <= b["made"] <= b["made_high"] for b in bins)
    assert bins[0]["predicted"] < bins[-1]["predicted"]


def test_accuracy_uses_a_fixed_half_threshold():
    preds = pd.DataFrame(
        {
            "shot_id": ["a", "b", "c", "d"],
            "game_id": ["g"] * 4,
            "cohort": ["c"] * 4,
            "split": ["test"] * 4,
            "model": ["m"] * 4,
            "y_true": [1, 0, 1, 0],
            "p_make": [0.5, 0.49, 0.2, 0.9],
        }
    )
    # 0.5 counts as a predicted make; only the first two are right.
    assert score_predictions(preds)[0]["accuracy"] == 0.5
