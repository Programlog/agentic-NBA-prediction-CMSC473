"""Tests for frozen experiment metrics verification."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from nba_shots.frozen_metrics_verification import (
    VerificationError,
    verify_all,
    verify_h2,
)

ROOT = Path(__file__).resolve().parents[1]
H2_RELATIVE = Path("experiments") / "h2" / "02_final_test_100" / "report"


def test_committed_frozen_results_match():
    result = verify_all(ROOT)
    assert result["H1_model_evaluations"] == 8
    assert result["H2_model_evaluations"] == 6


def make_h2_copy(tmp_path):
    destination = tmp_path / H2_RELATIVE
    destination.mkdir(parents=True)

    for filename in ("predictions.csv", "combined_metrics.csv"):
        shutil.copy2(ROOT / H2_RELATIVE / filename, destination / filename)
    return destination


def test_modified_prediction_is_detected(tmp_path):
    directory = make_h2_copy(tmp_path)
    path = directory / "predictions.csv"
    frame = pd.read_csv(path)

    index = frame.index[0]
    frame.loc[index, "p_make"] = 0.99
    frame.to_csv(path, index=False)

    with pytest.raises(VerificationError, match="calculated"):
        verify_h2(tmp_path)


def test_missing_model_shot_is_detected(tmp_path):
    directory = make_h2_copy(tmp_path)
    path = directory / "predictions.csv"
    frame = pd.read_csv(path)

    frame = frame.drop(index=frame.index[0])
    frame.to_csv(path, index=False)

    with pytest.raises(VerificationError, match="different shot IDs"):
        verify_h2(tmp_path)


def test_incorrect_published_metric_is_detected(tmp_path):
    directory = make_h2_copy(tmp_path)
    path = directory / "combined_metrics.csv"
    frame = pd.read_csv(path)

    frame.loc[0, "brier_score"] += 0.05
    frame.to_csv(path, index=False)

    with pytest.raises(VerificationError, match="calculated"):
        verify_h2(tmp_path)
