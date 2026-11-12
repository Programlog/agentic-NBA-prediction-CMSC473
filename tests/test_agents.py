import importlib.util
import re

import numpy as np
import pandas as pd
import pytest

from nba_shots.agents import (
    EXPECTED_OUTPUT,
    FEATURE_DEFINITIONS,
    FEATURE_GUIDANCE,
    INSTRUCTIONS,
    PROMPT_VERSION,
    OllamaClient,
    call_budget,
    crewai_assessment,
    parse_assessment,
    predict_condition,
    prompt_features,
    select_evaluation_rows,
    summarize_agent_usage,
)
from nba_shots.common import ALLOWED_FEATURES
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
        {
            "shot_distance": 20,
            "period": 1,
            "player_prior_make_rate": None,
        },
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


@pytest.mark.skipif(
    importlib.util.find_spec("crewai") is None,
    reason="optional agents extra",
)
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
    result = crewai_assessment(
        client,
        "Context analyst",
        {"shot_distance": 22},
        [],
    )

    assert result["p_make"] == 0.4 and len(client.calls) == 1


@pytest.mark.skipif(
    importlib.util.find_spec("crewai") is None,
    reason="optional agents extra",
)
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
                    "message": {"content": '{"p_make": 0.4, "rationale": "Test"}'},
                    "prompt_eval_count": 100,
                    "eval_count": 20,
                }
            )

    monkeypatch.setattr(agents.requests, "Session", Session)

    source = tmp_path / "shots.csv"
    raw_shots.to_csv(source, index=False)

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

    # Missing player history reduces the adaptive call budget:
    # 1 single + 3 iterative + 1 decomposed
    # + 3 ensemble + 3 multi = 11 calls.
    assert result["actual_calls_this_run"] == 11
    assert len(requests_seen) == 11

    assert all(m["coverage"] == 1 for m in result["metrics"])

    usage = pd.read_csv(result["usage_report"])

    assert set(usage["condition"]) == {
        "llm_single",
        "llm_single_budget",
        "llm_single_decomposed",
        "llm_single_ensemble",
        "llm_multi",
    }

    assert usage["calls_this_run"].sum() == 11

    sent = [m["content"] for request in requests_seen for m in request["messages"]]

    assert any('"features": {"' in text for text in sent)
    assert not any('"features": {}' in text for text in sent)

    resumed = agents.run_agents(
        shots,
        tmp_path / "run",
        limit=1,
    )

    assert resumed["actual_calls_this_run"] == 0
    assert len(requests_seen) == 11

    with pytest.raises(ValueError, match="Experiment changed"):
        agents.run_agents(
            shots,
            tmp_path / "run",
            limit=2,
        )


def test_prompt_text_contains_no_sample_probability():
    assert not re.search(r"\d", EXPECTED_OUTPUT)


def test_single_decomposed_has_full_input_one_call_and_no_prior_assessments():
    seen = []

    def assessor(role, features, previous):
        seen.append((role, features.copy(), list(previous)))
        return {"p_make": 0.35, "rationale": "Test"}

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
    assert call_budget(features, "llm_single_decomposed") == 1
    assert len(seen) == 1

    role, received, previous = seen[0]

    assert "game context" in role
    assert "player history" in role
    assert "defense" in role
    assert received == features
    assert previous == []


def test_ensemble_averages_all_independent_calls_without_feedback():
    seen = []
    values = iter([0.1, 0.2, 0.6])

    def assessor(role, features, previous):
        seen.append((role, features.copy(), list(previous)))
        return {"p_make": next(values), "rationale": "Test"}

    features = {
        "period": 4,
        "shot_distance": 18,
    }

    result = predict_condition(
        features,
        "llm_single_ensemble",
        assessor,
    )

    assert call_budget(features, "llm_single_ensemble") == 3
    assert result["p_make"] == pytest.approx(0.3)
    assert len(seen) == 3
    assert all(received == features and previous == [] for _, received, previous in seen)


def test_all_conditions_reject_hidden_outcomes():
    features = {
        "shot_distance": 20,
        "y_true": 1,
    }

    def assessor(role, inputs, previous):
        raise AssertionError("Unsafe data reached assessor")

    for condition in [
        "llm_single",
        "llm_single_budget",
        "llm_single_decomposed",
        "llm_single_ensemble",
        "llm_multi",
    ]:
        with pytest.raises(ValueError, match="allowlisted"):
            predict_condition(features, condition, assessor)


def test_missing_history_reduces_multi_and_ensemble_call_budgets():
    features = {
        "period": 1,
        "shot_distance": 8,
        "player_prior_make_rate": None,
    }

    assert call_budget(features, "llm_single_ensemble") == 3
    assert call_budget(features, "llm_multi") == 3
    assert call_budget(features, "llm_single_decomposed") == 1


def test_usage_report_includes_failure_cost_and_cached_success():
    rows = pd.DataFrame(
        [
            {
                "model": "arm_a",
                "status": "ok",
                "cached": False,
                "calls": 2,
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "runtime_seconds": 1.2,
            },
            {
                "model": "arm_a",
                "status": "error",
                "cached": False,
                "calls": 1,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "runtime_seconds": 0.4,
            },
            {
                "model": "arm_b",
                "status": "ok",
                "cached": True,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "runtime_seconds": 0.01,
            },
        ]
    )

    report = summarize_agent_usage(rows).set_index("condition")

    assert report.loc["arm_a", "shots_attempted"] == 2
    assert report.loc["arm_a", "shots_failed"] == 1
    assert report.loc["arm_a", "coverage"] == 0.5
    assert report.loc["arm_a", "calls_this_run"] == 3
    assert report.loc["arm_a", "prompt_tokens_this_run"] == 120
    assert report.loc["arm_a", "runtime_seconds_this_run"] == pytest.approx(1.6)

    assert report.loc["arm_b", "cache_hits"] == 1
    assert report.loc["arm_b", "calls_this_run"] == 0


# Reject invalid measurements instead of silently
# producing misleading compute-usage summaries.
@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("calls", -1),
        ("prompt_tokens", np.nan),
        ("completion_tokens", -5),
        ("runtime_seconds", np.nan),
        ("runtime_seconds", -0.5),
    ],
)
def test_usage_report_rejects_invalid_compute_measurements(field, invalid_value):
    row = {
        "model": "llm_single",
        "status": "ok",
        "cached": False,
        "calls": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "runtime_seconds": 1.5,
    }

    row[field] = invalid_value

    with pytest.raises(ValueError, match="Invalid usage"):
        summarize_agent_usage(pd.DataFrame([row]))


def test_feature_definitions_cover_all_allowed_inputs():
    assert set(FEATURE_DEFINITIONS) == ALLOWED_FEATURES

    for name, description in FEATURE_DEFINITIONS.items():
        assert description.strip()
        assert f"{name}: {description}" in FEATURE_GUIDANCE


def test_shared_prompt_explains_ambiguous_features():
    assert FEATURE_GUIDANCE in INSTRUCTIONS

    assert "H means home and A means away" in INSTRUCTIONS
    assert "not the shooter's position on the court" in INSTRUCTIONS

    assert "Seconds remaining on the possession shot clock" in INSTRUCTIONS
    assert "a higher value means more time remains" in INSTRUCTIONS

    assert "not the importance of the attempt" in INSTRUCTIONS
    assert "not physical distance in feet" in INSTRUCTIONS
    assert "Missing or null measurements are unknown, not zero" in INSTRUCTIONS


def test_feature_guidance_does_not_expand_allowed_inputs():
    assert "shot_number" not in FEATURE_DEFINITIONS
    assert "game_id" not in FEATURE_DEFINITIONS
    assert "player_id" not in FEATURE_DEFINITIONS
    assert "y_true" not in FEATURE_DEFINITIONS

    row = {
        "shot_clock": 18.7,
        "location": "A",
        "shot_value": 2,
        "y_true": 1,
        "player_id": "hidden",
    }

    assert prompt_features(row) == {
        "location": "A",
        "shot_clock": 18.7,
        "shot_value": 2,
    }


def test_feature_semantics_use_new_prompt_version():
    assert PROMPT_VERSION == "release-v4-feature-semantics"
