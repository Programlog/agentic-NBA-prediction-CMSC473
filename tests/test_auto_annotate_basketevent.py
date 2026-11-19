import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts" / "auto_annotate_basketevent.py"
spec = importlib.util.spec_from_file_location("auto_annotate_basketevent", HERE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_xywh_to_xyxy():
    assert mod.xywh_to_xyxy([10, 20, 30, 40], 100, 100) == [10, 20, 40, 60]
    assert mod.xywh_to_xyxy([90, 90, 20, 20], 100, 100) == [90, 90, 100, 100]
    assert mod.xywh_to_xyxy([10, 20, -5, 10], 100, 100) is None


def test_release_candidates_with_shooter_ball_departure():
    n = 30
    shooter = [[100, 100, 200, 300] for _ in range(n)]
    ball = []
    for i in range(n):
        if i < 12:
            ball.append([145, 145, 155, 155])
        else:
            y = 145 - 16 * (i - 11)
            ball.append([145, y, 155, y + 10])
    result = mod.ball_candidates(shooter, ball, n)
    assert result["candidates"]
    assert abs(result["candidates"][0] - 12) <= 5


def test_missing_ball_has_no_candidate():
    n = 25
    assert mod.ball_candidates([[10, 10, 100, 200]] * n, [None] * n, n)["candidates"] == []


def test_nearest_different_jersey_is_only_a_proposal():
    n = 10
    data = {
        "player_0": {"jersey_color": "white", "trajectory": [[100, 100, 50, 80]] * n},
        "player_1": {"jersey_color": "black", "trajectory": [[160, 100, 50, 80]] * n},
        "player_2": {"jersey_color": "white", "trajectory": [[110, 110, 50, 80]] * n},
    }
    role, box, reason = mod.defender_proposal(data, "player_0", 5, 1280, 720, n)
    assert role == "player_1"
    assert box == [160, 100, 210, 180]
    assert reason.endswith("unverified")


def test_same_color_and_far_opponents_not_assigned():
    n = 10
    data = {
        "player_0": {"jersey_color": "white", "trajectory": [[100, 100, 50, 80]] * n},
        "player_1": {"jersey_color": "white", "trajectory": [[120, 100, 50, 80]] * n},
    }
    assert mod.defender_proposal(data, "player_0", 5, 1280, 720, n)[0] is None
    data["player_1"]["jersey_color"] = "black"
    data["player_1"]["trajectory"] = [[1100, 500, 50, 80]] * n
    assert mod.defender_proposal(data, "player_0", 5, 1280, 720, n)[0] is None
