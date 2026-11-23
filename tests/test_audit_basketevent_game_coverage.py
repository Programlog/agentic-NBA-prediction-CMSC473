"""Offline tests for BasketEvent event-metadata coverage accounting."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "audit_basketevent_game_coverage.py"


@pytest.fixture(scope="module")
def audit_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec = importlib.util.spec_from_file_location("h1_coverage_audit", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPT_DIR))


def test_missing_event_fields(audit_module):
    data = {
        "player_0": {"trajectory": [], "player_name": "A"},
        "player_1": {"trajectory": []},
    }

    fields, non_null, actions = audit_module.count_event_metadata(data)

    assert fields == 0
    assert non_null == 0
    assert dict(actions) == {}


def test_null_event_fields(audit_module):
    data = {
        "player_0": {"event": None, "trajectory": []},
        "player_1": {"event": None, "trajectory": []},
    }

    fields, non_null, actions = audit_module.count_event_metadata(data)

    assert fields == 2
    assert non_null == 0
    assert dict(actions) == {}


def test_valid_event_dictionaries(audit_module):
    data = {
        "player_0": {"event": {"actionType": "Made Shot"}},
        "player_1": {"event": {"actionType": "Missed Shot"}},
        "player_2": {"event": None},
        "player_3": {"trajectory": []},
    }

    fields, non_null, actions = audit_module.count_event_metadata(data)

    assert fields == 3
    assert non_null == 2
    assert actions == {"Made Shot": 1, "Missed Shot": 1}


def test_unexpected_annotation_and_event_types(audit_module):
    data = {
        "player_0": {"event": "Made Shot"},
        "player_1": {"event": {"actionType": "Foul"}},
        "player_2": [],
        "player_3": None,
        "player_4": {"event": {}},
    }

    fields, non_null, actions = audit_module.count_event_metadata(data)

    assert fields == 3
    assert non_null == 2
    assert actions == {"Foul": 1, "<missing>": 1}
