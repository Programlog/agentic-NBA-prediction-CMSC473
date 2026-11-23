import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pytest

from nba_shots.agents import (
    CONDITIONS,
    EXPECTED_OUTPUT,
    FEATURE_DEFINITIONS,
    FEATURE_GUIDANCE,
    INSTRUCTIONS,
    PROMPT_VERSION,
    OllamaClient,
    _load_assessment_cache,
    call_budget,
    crewai_assessment,
    parse_assessment,
    predict_condition,
    prompt_features,
    select_evaluation_rows,
)
from nba_shots.common import ALLOWED_FEATURES
from nba_shots.data import (
    assign_splits,
    normalize_kaggle,
)


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
    assert all(call[1] == features for call in calls)

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

    examples = [
        {
            "shot_distance": 20,
            "period": 1,
        },
        {
            "shot_distance": 20,
            "period": 1,
            "player_prior_make_rate": None,
        },
    ]

    for features in examples:
        calls.clear()

        predict_condition(
            features,
            "llm_multi",
            assessor,
        )

        assert [call[0] for call in calls] == [
            "Game context analyst",
            "Shot mechanics and defense analyst",
            "Shot probability aggregator",
        ]

        assert all(call[1] for call in calls)
        assert len(calls[2][2]) == 2

        calls.clear()

        predict_condition(
            features,
            "llm_single_budget",
            assessor,
        )

        assert len(calls) == 3
        assert (
            call_budget(
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
            assert url == ("http://127.0.0.1:11434/api/tags")

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
            assert url == ("http://127.0.0.1:11434/api/chat")

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

    # Missing player history:
    # 1 single + 3 budget + 1 decomposed + 3 ensemble + 3 multi.
    assert result["actual_calls_this_run"] == 11
    assert len(requests_seen) == 11

    assert all(metric["coverage"] == 1 for metric in result["metrics"])

    ensemble_requests = [
        request for request in requests_seen if request["options"]["temperature"] == 0.7
    ]

    assert len(ensemble_requests) == 3

    assert [request["options"]["seed"] for request in ensemble_requests] == [42, 43, 44]

    assert sum(request["options"]["temperature"] == 0.0 for request in requests_seen) == 8

    sent = [message["content"] for request in requests_seen for message in request["messages"]]

    assert any('"features": {"' in text for text in sent)

    assert not any('"features": {}' in text for text in sent)

    resumed = agents.run_agents(
        shots,
        tmp_path / "run",
        limit=1,
    )

    assert resumed["actual_calls_this_run"] == 0
    assert len(requests_seen) == 11

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
        "prompt_version": PROMPT_VERSION,
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
            self.seed = 42
            self.temperature = 0.0

    def fake_assessment(
        client,
        role,
        features,
        previous,
    ):
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

    assert first["actual_calls_this_run"] == 11
    assert validate_artifacts(output)["valid"] is True

    fresh = pd.read_csv(output / "predictions.csv")

    assert fresh["original_cost_provenance"].eq("fresh_trace").all()

    assert (fresh["original_calls"] == fresh["calls"]).all()

    caches = list((output / "cache").glob("*.json"))

    assert len(caches) == 5

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
    assert cached["original_calls"].sum() == 11
    assert cached["original_prompt_tokens"].sum() == 1100
    assert cached["original_completion_tokens"].sum() == 220

    assert cached["original_cost_provenance"].eq("cache_v2").all()

    assert len((output / "traces.jsonl").read_text().splitlines()) == 5

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

    assert ledger["original_calls_verified"].sum() == 11


def test_decomposed_single_agent_uses_one_call():
    seen = []

    def assessor(role, features, previous):
        seen.append(
            (
                role,
                features.copy(),
                list(previous),
            )
        )

        return {
            "p_make": 0.35,
            "rationale": "Test",
        }

    features = {
        "period": 3,
        "shot_distance": 22,
        "player_prior_make_rate": 0.42,
    }

    result = predict_condition(
        features,
        "llm_single_decomposed",
        assessor,
    )

    assert result["p_make"] == 0.35
    assert (
        call_budget(
            features,
            "llm_single_decomposed",
        )
        == 1
    )

    assert len(seen) == 1

    role, received, previous = seen[0]

    assert "game context" in role
    assert "player history" in role
    assert "defense" in role
    assert received == features
    assert previous == []


def test_ensemble_averages_predictions_without_feedback():
    seen = []
    values = iter([0.1, 0.2, 0.6])

    def assessor(role, features, previous):
        seen.append(
            (
                role,
                features.copy(),
                list(previous),
            )
        )

        return {
            "p_make": next(values),
            "rationale": "Test",
        }

    features = {
        "period": 4,
        "shot_distance": 18,
    }

    result = predict_condition(
        features,
        "llm_single_ensemble",
        assessor,
    )

    assert (
        call_budget(
            features,
            "llm_single_ensemble",
        )
        == 3
    )

    assert result["p_make"] == pytest.approx(0.3)
    assert len(seen) == 3

    assert all(received == features and previous == [] for _, received, previous in seen)


def test_all_five_conditions_reject_hidden_outcomes():
    features = {
        "shot_distance": 20,
        "y_true": 1,
    }

    def assessor(role, inputs, previous):
        raise AssertionError("Outcome reached the assessor")

    for condition in CONDITIONS:
        with pytest.raises(
            ValueError,
            match="allowlisted",
        ):
            predict_condition(
                features,
                condition,
                assessor,
            )


def test_five_condition_adaptive_call_budgets():
    features = {
        "period": 1,
        "shot_distance": 8,
        "player_prior_make_rate": None,
    }

    expected = {
        "llm_single": 1,
        "llm_single_budget": 3,
        "llm_single_decomposed": 1,
        "llm_single_ensemble": 3,
        "llm_multi": 3,
    }

    assert {
        condition: call_budget(
            features,
            condition,
        )
        for condition in CONDITIONS
    } == expected

    assert sum(expected.values()) == 11


def test_feature_definitions_and_guidance_cover_allowed_inputs():
    assert set(FEATURE_DEFINITIONS) == ALLOWED_FEATURES

    assert all(description.strip() for description in FEATURE_DEFINITIONS.values())

    assert all(
        f"{name}: {description}" in FEATURE_GUIDANCE
        for name, description in FEATURE_DEFINITIONS.items()
    )

    assert FEATURE_GUIDANCE in INSTRUCTIONS
    assert "H means home and A means away" in INSTRUCTIONS
    assert "Missing or null measurements are unknown, not zero" in INSTRUCTIONS
    assert "y_true" not in FEATURE_DEFINITIONS
    assert PROMPT_VERSION == "release-v4-feature-semantics"


def test_integrated_balanced_selection_runs_all_five_conditions(
    monkeypatch,
    tmp_path,
):
    from nba_shots import agents

    rows = pd.DataFrame(
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

    source = tmp_path / "shots.parquet"
    source.write_bytes(b"synthetic-shots")

    monkeypatch.setattr(
        agents.pd,
        "read_parquet",
        lambda _: rows.copy(),
    )

    monkeypatch.setattr(
        agents,
        "validate_splits",
        lambda _: None,
    )

    monkeypatch.setattr(
        agents,
        "write_scores",
        lambda frame, *args, **kwargs: [
            {
                "coverage": float(
                    frame.loc[
                        frame["model"].eq(condition),
                        "status",
                    ]
                    .eq("ok")
                    .mean()
                )
            }
            for condition in agents.CONDITIONS
        ],
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
            self.digest = "test-digest"
            self.calls = []
            self.seed = seed
            self.temperature = 0.0

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

    monkeypatch.setattr(agents, "OllamaClient", FakeClient)
    monkeypatch.setattr(
        agents,
        "crewai_assessment",
        fake_assessment,
    )

    output = tmp_path / "balanced_run"

    result = agents.run_agents(
        source,
        output,
        limit=15,
        seed=42,
        games=5,
    )

    selection = pd.read_csv(output / "selection.csv")
    predictions = pd.read_csv(result["predictions"])

    assert len(selection) == 15
    assert selection["game_id"].nunique() == 5
    assert (selection.groupby("game_id").size() == 3).all()

    assert len(predictions) == 15 * 5
    assert predictions["status"].eq("ok").all()
    assert predictions["cache_key_sha256"].nunique() == len(predictions)

    for condition in agents.CONDITIONS:
        subset = predictions.loc[predictions["model"].eq(condition)]
        assert set(subset["shot_id"]) == set(selection["shot_id"])

    spec = json.loads((output / "experiment.json").read_text())["spec"]

    assert spec["sampling_method"] == "balanced_games_v1"
    assert spec["sampling_games"] == 5
    assert spec["shot_ids"] == selection["shot_id"].tolist()


def test_condition_subset_requires_core_conditions_and_keeps_canonical_order():
    from nba_shots import agents

    assert agents.registered_conditions(None) == agents.CONDITIONS
    assert agents.registered_conditions(["llm_multi", "llm_single", "llm_single_budget"]) == {
        "llm_single": 1,
        "llm_single_budget": 4,
        "llm_multi": 4,
    }

    with pytest.raises(ValueError, match="required conditions"):
        agents.registered_conditions(["llm_single", "llm_multi"])

    with pytest.raises(ValueError, match="Unknown agent condition"):
        agents.registered_conditions([*agents.REQUIRED_CONDITIONS, "llm_unknown"])

    with pytest.raises(ValueError, match="must not repeat"):
        agents.registered_conditions([*agents.REQUIRED_CONDITIONS, "llm_multi"])


def test_final_test_run_with_core_conditions_registers_only_those(monkeypatch, tmp_path):
    from nba_shots import agents

    rows = pd.DataFrame(
        [
            {
                "shot_id": f"shot-{game}-{shot}",
                "game_id": f"game-{game}",
                "cohort": "kaggle_2014_15",
                "split": "test",
                "y_true": shot % 2,
                "shot_distance": float(shot * 5 + 3),
                "period": 1,
            }
            for game in range(4)
            for shot in range(4)
        ]
    )
    source = tmp_path / "shots.parquet"
    source.write_bytes(b"synthetic-shots")
    monkeypatch.setattr(agents.pd, "read_parquet", lambda _: rows.copy())
    monkeypatch.setattr(agents, "validate_splits", lambda _: None)
    monkeypatch.setattr(agents, "write_scores", lambda *args, **kwargs: [])

    class FakeClient:
        def __init__(self, model, base_url, seed, num_predict):
            self.model = model
            self.digest = "test-digest"
            self.calls = []
            self.seed = seed
            self.temperature = 0.0

    def fake_assessment(client, role, features, previous):
        client.calls.append(
            {"status": "ok", "prompt_tokens": 10, "completion_tokens": 5, "runtime_seconds": 0.01}
        )
        return {"p_make": 0.5, "rationale": "Synthetic prediction"}

    monkeypatch.setattr(agents, "OllamaClient", FakeClient)
    monkeypatch.setattr(agents, "crewai_assessment", fake_assessment)

    core = list(agents.REQUIRED_CONDITIONS)

    with pytest.raises(ValueError, match="requires --final"):
        agents.run_agents(
            source, tmp_path / "unfrozen", split="test", limit=6, games=3, conditions=core
        )

    output = tmp_path / "final_run"
    result = agents.run_agents(
        source, output, split="test", final=True, limit=6, games=3, conditions=core
    )

    predictions = pd.read_csv(result["predictions"])
    spec = json.loads((output / "experiment.json").read_text())["spec"]

    assert spec["conditions"] == {"llm_single": 1, "llm_single_budget": 4, "llm_multi": 4}
    assert spec["split"] == "test"
    assert set(predictions["model"]) == set(core)
    assert len(predictions) == 6 * 3
    assert predictions["status"].eq("ok").all()
