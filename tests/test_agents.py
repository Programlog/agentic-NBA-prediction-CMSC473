import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pytest

from nba_shots.agents import (
    EXPECTED_OUTPUT,
    OllamaClient,
    call_budget,
    crewai_assessment,
    parse_assessment,
    predict_condition,
    prompt_features,
    select_evaluation_rows,
)
from nba_shots.data import assign_splits, normalize_kaggle


def test_prompts_exclude_labels_and_reidentifying_metadata():
    row = {
        "y_true": 1,
        "FGM": 1,
        "player_id": "123",
        "game_date": "2015-01-01",
        "game_id": "0021400001",
        "shot_id": "s",
        "shot_clock": np.nan,
        "shot_distance": np.float64(20),
        "location": "H",
        "clip_path": "made.mp4",
    }
    assert prompt_features(row) == {"shot_clock": None, "shot_distance": 20.0, "location": "H"}


def test_architecture_and_equal_call_control():
    calls = []

    def assessor(role, features, previous):
        calls.append((role, features.copy(), list(previous)))
        return {"p_make": 0.5, "rationale": "Test evidence"}

    features = {
        "shot_distance": 20,
        "period": 1,
        "player_prior_make_rate": 0.4,
        "elbow_angle_deg": 90,
    }
    predict_condition(features, "llm_single", assessor)
    assert len(calls) == 1 and calls[0][1] == features
    calls.clear()
    predict_condition(features, "llm_single_budget", assessor)
    assert len(calls) == 4 and all(c[1] == features for c in calls)
    calls.clear()
    predict_condition(features, "llm_multi", assessor)
    assert len(calls) == 4
    assert calls[0][1] == {"period": 1}
    assert calls[1][1] == {"player_prior_make_rate": 0.4}
    assert calls[3][1] == features and len(calls[3][2]) == 3
    with pytest.raises(ValueError, match="allowlisted"):
        predict_condition({"y_true": 1}, "llm_single", assessor)


def test_specialist_without_inputs_is_skipped_and_budget_matches():
    calls = []

    def assessor(role, features, previous):
        calls.append((role, features.copy(), list(previous)))
        return {"p_make": 0.5, "rationale": "Test evidence"}

    for features in [
        {"shot_distance": 20, "period": 1},
        {"shot_distance": 20, "period": 1, "player_prior_make_rate": None},
    ]:
        calls.clear()
        predict_condition(features, "llm_multi", assessor)
        assert [c[0] for c in calls] == [
            "Game context analyst",
            "Shot mechanics and defense analyst",
            "Shot probability aggregator",
        ]
        assert all(c[1] for c in calls) and len(calls[2][2]) == 2
        calls.clear()
        predict_condition(features, "llm_single_budget", assessor)
        assert len(calls) == call_budget(features, "llm_multi") == 3


@pytest.mark.parametrize(
    "raw",
    [
        '{"p_make": 1.1, "rationale": "bad"}',
        '{"p_make": true, "rationale": "bad"}',
        '{"p_make": "0.5", "rationale": "bad"}',
        '{"p_make": 0.5}',
    ],
)
def test_bad_llm_outputs_are_errors(raw):
    with pytest.raises(ValueError):
        parse_assessment(raw)


def test_selection_is_label_blind_and_capped(raw_shots):
    shots, _ = normalize_kaggle(raw_shots)
    shots, _ = assign_splits(shots)
    first = select_evaluation_rows(shots, "validation", 4)
    shots.y_true = 1 - shots.y_true
    second = select_evaluation_rows(shots, "validation", 4)
    assert first.shot_id.tolist() == second.shot_id.tolist()
    with pytest.raises(ValueError, match="500"):
        select_evaluation_rows(shots, "validation", 501)


def test_remote_model_endpoints_are_rejected_before_network_access():
    with pytest.raises(ValueError, match="local Ollama"):
        OllamaClient("model", "https://api.example.com")


@pytest.mark.skipif(importlib.util.find_spec("crewai") is None, reason="optional agents extra")
def test_actual_crewai_adapter_obeys_one_call_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("CREWAI_TESTING", "true")

    class FakeLocalClient:
        model = "local-test"

        def __init__(self):
            self.calls = []

        def call(self, messages):
            self.calls.append(messages)
            return {"p_make": 0.4, "rationale": "A long contested shot"}

    client = FakeLocalClient()
    result = crewai_assessment(client, "Context analyst", {"shot_distance": 22}, [])
    assert result["p_make"] == 0.4 and len(client.calls) == 1


@pytest.mark.skipif(importlib.util.find_spec("crewai") is None, reason="optional agents extra")
def test_complete_agent_run_uses_local_payloads_and_resumes_cache(raw_shots, monkeypatch, tmp_path):
    import json

    from nba_shots import agents
    from nba_shots.common import load_config
    from nba_shots.data import prepare_csv

    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "state"))
    requests_seen = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Session:
        def get(self, url, **kwargs):
            assert url == "http://127.0.0.1:11434/api/tags"
            return Response({"models": [{"name": "qwen3:4b-instruct", "digest": "test-digest"}]})

        def post(self, url, **kwargs):
            assert url == "http://127.0.0.1:11434/api/chat"
            payload = kwargs["json"]
            encoded = json.dumps(payload)
            assert "y_true" not in encoded and "game_id" not in encoded
            assert payload["options"]["num_predict"] == 512
            requests_seen.append(payload)
            return Response(
                {
                    "message": {"content": '{"p_make": 0.4, "rationale": "Test"}'},
                    "prompt_eval_count": 100,
                    "eval_count": 20,
                }
            )

    monkeypatch.setattr(agents.requests, "Session", Session)
    source = tmp_path / "shots.csv"
    raw_shots.to_csv(source, index=False)
    shots = prepare_csv(source, tmp_path / "prepared", load_config())
    result = agents.run_agents(shots, tmp_path / "run", limit=1)
    # Default config has no player history, so the history specialist is skipped: 1 + 3 + 3.
    assert result["actual_calls_this_run"] == 7 and len(requests_seen) == 7
    assert all(m["coverage"] == 1 for m in result["metrics"])
    sent = [m["content"] for r in requests_seen for m in r["messages"]]
    assert any('"features": {"' in text for text in sent)
    assert not any('"features": {}' in text for text in sent)
    resumed = agents.run_agents(shots, tmp_path / "run", limit=1)
    assert resumed["actual_calls_this_run"] == 0 and len(requests_seen) == 7
    with pytest.raises(ValueError, match="Experiment changed"):
        agents.run_agents(shots, tmp_path / "run", limit=2)


def test_prompt_text_contains_no_sample_probability():
    assert not re.search(r"\d", EXPECTED_OUTPUT)


def sampling_shots():
    """Six games, five shots each, for deterministic sampling tests."""
    return pd.DataFrame(
        [
            {
                "shot_id": f"shot-{game}-{shot}",
                "game_id": f"game-{game}",
                "cohort": "kaggle_2014_15",
                "split": "validation",
                "y_true": shot % 2,
                "shot_distance": float(shot * 5 + 3),
                "period": 1,
            }
            for game in range(6)
            for shot in range(5)
        ]
    )


def test_balanced_agent_selection_is_reproducible():
    rows = sampling_shots()

    first = select_evaluation_rows(
        rows,
        "validation",
        15,
        seed=42,
        games=5,
    )

    shuffled = rows.sample(
        frac=1,
        random_state=11,
    )

    second = select_evaluation_rows(
        shuffled,
        "validation",
        15,
        seed=42,
        games=5,
    )

    assert first["shot_id"].tolist() == (second["shot_id"].tolist())

    counts = first.groupby("game_id").size()

    assert len(counts) == 5
    assert (counts == 3).all()


def test_existing_agent_selection_still_supported():
    rows = sampling_shots()

    original = select_evaluation_rows(
        rows,
        "validation",
        10,
        seed=42,
    )

    backward_compatible = select_evaluation_rows(
        rows,
        "validation",
        10,
        seed=42,
        games=None,
    )

    assert original["shot_id"].tolist() == (backward_compatible["shot_id"].tolist())


def test_agent_runner_uses_balanced_sampling(
    monkeypatch,
    tmp_path,
):
    """Verify the real runner saves the balanced shot IDs."""
    from nba_shots import agents

    rows = sampling_shots()
    source = tmp_path / "shots.parquet"
    rows.to_parquet(source, index=False)

    # The fixture contains only validation shots; this test isolates
    # the sampling-to-runner wiring from full chronological splitting.
    monkeypatch.setattr(
        agents,
        "validate_splits",
        lambda frame: None,
    )

    class FakeClient:
        def __init__(
            self,
            model,
            base_url,
            seed,
            num_predict,
        ):
            self.model = model
            self.digest = "fake-model-digest"
            self.calls = []

    def fake_assessment(client, role, features, previous):
        client.calls.append(
            {
                "status": "ok",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "runtime_seconds": 0.01,
            }
        )

        return {
            "p_make": 0.5,
            "rationale": "Synthetic prediction",
        }

    monkeypatch.setattr(
        agents,
        "OllamaClient",
        FakeClient,
    )

    monkeypatch.setattr(
        agents,
        "crewai_assessment",
        fake_assessment,
    )

    expected = select_evaluation_rows(
        rows,
        "validation",
        15,
        seed=42,
        games=5,
    )

    output = tmp_path / "balanced_run"

    result = agents.run_agents(
        source,
        output,
        limit=15,
        seed=42,
        games=5,
    )

    actual = pd.read_csv(output / "selection.csv")

    assert actual["shot_id"].tolist() == (expected["shot_id"].tolist())

    assert actual["game_id"].nunique() == 5

    assert (actual.groupby("game_id").size() == 3).all()

    spec = json.loads((output / "experiment.json").read_text())["spec"]

    assert spec["sampling_method"] == "balanced_games_v1"
    assert spec["sampling_games"] == 5
    assert spec["shot_ids"] == expected["shot_id"].tolist()

    predictions = pd.read_csv(result["predictions"])

    assert len(predictions) == 15 * len(agents.CONDITIONS)
    assert predictions["status"].eq("ok").all()

    for _, group in predictions.groupby("model"):
        assert set(group["shot_id"]) == set(expected["shot_id"])
