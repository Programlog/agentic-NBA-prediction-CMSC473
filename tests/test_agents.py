import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pytest

from nba_shots.agents import (
    EXPECTED_OUTPUT,
    OllamaClient,
    _load_assessment_cache,
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

    assert prompt_features(row) == {
        "shot_clock": None,
        "shot_distance": 20.0,
        "location": "H",
    }


def test_architecture_and_equal_call_control():
    calls = []

    def assessor(role, features, previous):
        calls.append((role, features.copy(), list(previous)))
        return {
            "p_make": 0.5,
            "rationale": "Test evidence",
        }

    features = {
        "shot_distance": 20,
        "period": 1,
        "player_prior_make_rate": 0.4,
        "elbow_angle_deg": 90,
    }

    predict_condition(
        features,
        "llm_single",
        assessor,
    )
    assert len(calls) == 1
    assert calls[0][1] == features

    calls.clear()
    predict_condition(
        features,
        "llm_single_budget",
        assessor,
    )
    assert len(calls) == 4
    assert all(c[1] == features for c in calls)

    calls.clear()
    predict_condition(
        features,
        "llm_multi",
        assessor,
    )
    assert len(calls) == 4
    assert calls[0][1] == {"period": 1}
    assert calls[1][1] == {"player_prior_make_rate": 0.4}
    assert calls[3][1] == features
    assert len(calls[3][2]) == 3

    with pytest.raises(
        ValueError,
        match="allowlisted",
    ):
        predict_condition(
            {"y_true": 1},
            "llm_single",
            assessor,
        )


def test_specialist_without_inputs_is_skipped_and_budget_matches():
    calls = []

    def assessor(role, features, previous):
        calls.append((role, features.copy(), list(previous)))
        return {
            "p_make": 0.5,
            "rationale": "Test evidence",
        }

    for features in [
        {
            "shot_distance": 20,
            "period": 1,
        },
        {
            "shot_distance": 20,
            "period": 1,
            "player_prior_make_rate": None,
        },
    ]:
        calls.clear()

        predict_condition(
            features,
            "llm_multi",
            assessor,
        )

        assert [c[0] for c in calls] == [
            "Game context analyst",
            "Shot mechanics and defense analyst",
            "Shot probability aggregator",
        ]

        assert all(c[1] for c in calls)
        assert len(calls[2][2]) == 2

        calls.clear()

        predict_condition(
            features,
            "llm_single_budget",
            assessor,
        )

        assert (
            len(calls)
            == call_budget(
                features,
                "llm_multi",
            )
            == 3
        )


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

    first = select_evaluation_rows(
        shots,
        "validation",
        4,
    )

    shots.y_true = 1 - shots.y_true

    second = select_evaluation_rows(
        shots,
        "validation",
        4,
    )

    assert first.shot_id.tolist() == second.shot_id.tolist()

    with pytest.raises(
        ValueError,
        match="500",
    ):
        select_evaluation_rows(
            shots,
            "validation",
            501,
        )


def test_remote_model_endpoints_are_rejected_before_network_access():
    with pytest.raises(
        ValueError,
        match="local Ollama",
    ):
        OllamaClient(
            "model",
            "https://api.example.com",
        )


@pytest.mark.skipif(
    importlib.util.find_spec("crewai") is None,
    reason="optional agents extra",
)
def test_actual_crewai_adapter_obeys_one_call_budget(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "CREWAI_STORAGE_DIR",
        str(tmp_path),
    )
    monkeypatch.setenv(
        "CREWAI_TESTING",
        "true",
    )

    class FakeLocalClient:
        model = "local-test"

        def __init__(self):
            self.calls = []

        def call(self, messages):
            self.calls.append(messages)
            return {
                "p_make": 0.4,
                "rationale": "A long contested shot",
            }

    client = FakeLocalClient()

    result = crewai_assessment(
        client,
        "Context analyst",
        {"shot_distance": 22},
        [],
    )

    assert result["p_make"] == 0.4
    assert len(client.calls) == 1


@pytest.mark.skipif(
    importlib.util.find_spec("crewai") is None,
    reason="optional agents extra",
)
def test_complete_agent_run_uses_local_payloads_and_resumes_cache(
    raw_shots,
    monkeypatch,
    tmp_path,
):
    from nba_shots import agents
    from nba_shots.common import load_config
    from nba_shots.data import prepare_csv

    monkeypatch.setenv(
        "CREWAI_STORAGE_DIR",
        str(tmp_path / "state"),
    )

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
            return Response(
                {
                    "models": [
                        {
                            "name": "qwen3:4b-instruct",
                            "digest": "test-digest",
                        }
                    ]
                }
            )

        def post(self, url, **kwargs):
            assert url == "http://127.0.0.1:11434/api/chat"

            payload = kwargs["json"]
            encoded = json.dumps(payload)

            assert "y_true" not in encoded
            assert "game_id" not in encoded
            assert payload["options"]["num_predict"] == 512

            requests_seen.append(payload)

            return Response(
                {
                    "message": {"content": ('{"p_make": 0.4, "rationale": "Test"}')},
                    "prompt_eval_count": 100,
                    "eval_count": 20,
                }
            )

    monkeypatch.setattr(
        agents.requests,
        "Session",
        Session,
    )

    source = tmp_path / "shots.csv"
    raw_shots.to_csv(
        source,
        index=False,
    )

    shots = prepare_csv(
        source,
        tmp_path / "prepared",
        load_config(),
    )

    result = agents.run_agents(
        shots,
        tmp_path / "run",
        limit=1,
    )

    assert result["actual_calls_this_run"] == 7
    assert len(requests_seen) == 7
    assert all(m["coverage"] == 1 for m in result["metrics"])

    sent = [message["content"] for request in requests_seen for message in request["messages"]]

    assert any('"features": {"' in text for text in sent)

    assert not any('"features": {}' in text for text in sent)

    resumed = agents.run_agents(
        shots,
        tmp_path / "run",
        limit=1,
    )

    assert resumed["actual_calls_this_run"] == 0
    assert len(requests_seen) == 7

    with pytest.raises(
        ValueError,
        match="Experiment changed",
    ):
        agents.run_agents(
            shots,
            tmp_path / "run",
            limit=2,
        )


def test_prompt_text_contains_no_sample_probability():
    assert not re.search(
        r"\d",
        EXPECTED_OUTPUT,
    )


def test_v2_cache_round_trip_validates_identity_and_usage(tmp_path):
    key = "a" * 64

    identity = {
        "shot_id": "s1",
        "condition": "llm_single",
        "model": "qwen3:4b-instruct",
        "model_digest": "local",
        "prompt_version": "release-v2",
        "implementation_sha256": "sha",
    }

    payload = {
        "schema_version": 2,
        "cache_key_sha256": key,
        "identity": identity,
        "assessment": {
            "p_make": 0.55,
            "rationale": "context",
        },
        "original_usage": {
            "calls": 1,
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "runtime_seconds": 4.2,
        },
    }

    cache = tmp_path / f"{key}.json"

    cache.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assessment, usage = _load_assessment_cache(
        cache,
        key,
        identity,
        1,
    )

    assert assessment["p_make"] == 0.55
    assert usage == payload["original_usage"]

    with pytest.raises(
        ValueError,
        match="identity",
    ):
        _load_assessment_cache(
            cache,
            key,
            {
                **identity,
                "model_digest": "wrong",
            },
            1,
        )

    with pytest.raises(
        ValueError,
        match="budget",
    ):
        _load_assessment_cache(
            cache,
            key,
            identity,
            3,
        )

    payload["original_usage"]["prompt_tokens"] = -1

    cache.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid nonnegative",
    ):
        _load_assessment_cache(
            cache,
            key,
            identity,
            1,
        )


def test_legacy_assessment_cache_has_no_original_usage(tmp_path):
    cache = tmp_path / "legacy.json"

    cache.write_text(
        json.dumps(
            {
                "p_make": 0.5,
                "rationale": "old",
            }
        ),
        encoding="utf-8",
    )

    assessment, usage = _load_assessment_cache(
        cache,
        "0" * 64,
        {},
        1,
    )

    assert assessment["p_make"] == 0.5
    assert usage is None


def test_ledger_fresh_then_cached_execution_without_duplicate_traces(
    monkeypatch,
    tmp_path,
):
    from nba_shots import agents
    from nba_shots.experiment_summary import summarize_experiment
    from nba_shots.experiment_validation import validate_artifacts

    parquet = tmp_path / "shots.parquet"
    parquet.write_bytes(b"fixture checksum only")

    shots = pd.DataFrame(
        [
            {
                "shot_id": "shot-1",
                "game_id": "game-1",
                "cohort": "kaggle_2014_15",
                "split": "validation",
                "y_true": 1,
                "shot_distance": 14.0,
                "period": 2,
            }
        ]
    )

    monkeypatch.setattr(
        agents.pd,
        "read_parquet",
        lambda _: shots.copy(),
    )

    monkeypatch.setattr(
        agents,
        "validate_splits",
        lambda _: None,
    )

    monkeypatch.setattr(
        agents,
        "write_scores",
        lambda rows, *args, **kwargs: [
            {"coverage": float(rows[rows["model"] == arm]["status"].eq("ok").mean())}
            for arm in agents.CONDITIONS
        ],
    )

    class FakeClient:
        def __init__(self, *args):
            self.calls = []
            self.digest = "digest"

    def fake_assessment(client, role, features, previous):
        client.calls.append(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "runtime_seconds": 0.001,
                "status": "ok",
            }
        )

        return {
            "p_make": 0.42,
            "rationale": "test",
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

    output = tmp_path / "run"

    first = agents.run_agents(
        parquet,
        output,
        limit=1,
    )

    assert first["actual_calls_this_run"] == 7
    assert validate_artifacts(output)["valid"] is True

    fresh = pd.read_csv(output / "predictions.csv")

    assert fresh["original_cost_provenance"].eq("fresh_trace").all()

    assert (fresh["original_calls"] == fresh["calls"]).all()

    caches = list((output / "cache").glob("*.json"))

    assert len(caches) == 3
    assert all(json.loads(cache.read_text())["schema_version"] == 2 for cache in caches)

    second = agents.run_agents(
        parquet,
        output,
        limit=1,
    )

    assert second["actual_calls_this_run"] == 0
    assert validate_artifacts(output)["valid"] is True

    cached = pd.read_csv(output / "predictions.csv")

    assert cached["cached"].all()
    assert cached["calls"].eq(0).all()
    assert cached["original_calls"].sum() == 7
    assert cached["original_prompt_tokens"].sum() == 700
    assert cached["original_completion_tokens"].sum() == 140

    assert cached["original_cost_provenance"].eq("cache_v2").all()

    assert len((output / "traces.jsonl").read_text().splitlines()) == 3

    archives = list((output / "execution_history").iterdir())

    assert len(archives) == 1
    assert (archives[0] / "traces.jsonl").is_file()

    assert (archives[0] / "experiment.json").is_file()

    summary = summarize_experiment(
        output,
        tmp_path / "report",
    )

    assert summary["strict_validation_passed"] is True

    ledger = pd.read_csv(tmp_path / "report" / "cost_provenance.csv")

    assert ledger["original_calls_verified"].sum() == 7
