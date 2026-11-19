"""Focused regression checks for the opt-in disagreement-aware condition."""
import pytest

from nba_shots.agents import (
    CONDITIONS,
    OPTIONAL_CONDITIONS,
    call_budget,
    disagreement_aware_role,
    predict_condition,
    registered_conditions,
)


def test_default_arms_unchanged():
    assert len(CONDITIONS) == 5
    assert registered_conditions() == CONDITIONS
    assert 'llm_multi_disagreement' not in registered_conditions()
    assert OPTIONAL_CONDITIONS['llm_multi_disagreement'] == 4


def test_opt_in_registration():
    requested = ('llm_single', 'llm_single_budget', 'llm_multi', 'llm_multi_disagreement')
    assert 'llm_multi_disagreement' in registered_conditions(requested)


def test_role_checks_probabilities():
    role = disagreement_aware_role([{'p_make': 0.2}, {'p_make': 0.8}])
    assert '0.6000' in role
    with pytest.raises(ValueError):
        disagreement_aware_role([{'p_make': float('nan')}])


def test_condition_uses_specialists_and_aggregator():
    calls = []

    def assessor(role, features, previous):
        calls.append((role, features, previous))
        return {'p_make': 0.4 if not previous else 0.6, 'rationale': 'example'}

    features = {'shot_distance': 12.0, 'period': 2}
    assert call_budget(features, 'llm_multi_disagreement') == 3
    result = predict_condition(features, 'llm_multi_disagreement', assessor)
    assert result['p_make'] == 0.6
    assert len(calls) == 3
    assert len(calls[-1][2]) == 2
    assert 'Disagreement-aware' in calls[-1][0]
