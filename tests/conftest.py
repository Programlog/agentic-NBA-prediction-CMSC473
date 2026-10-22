import pandas as pd
import pytest


@pytest.fixture
def raw_shots():
    rows = []
    for day in range(1, 13):
        for game in range(2):
            for shot in range(4):
                rows.append(
                    {
                        "GAME_ID": str(21400000 + day * 2 + game),
                        "MATCHUP": f"NOV {day:02}, 2014 - CHA @ BKN",
                        "LOCATION": "H" if game else "A",
                        "GAME_CLOCK": "5:30",
                        "SHOT_RESULT": "made" if shot % 2 else "missed",
                        "player_id": str(100 + shot % 2),
                        "SHOT_NUMBER": str(shot + 1),
                        "SHOT_DIST": 3 + shot * 6,
                        "CLOSE_DEF_DIST": 2 + shot,
                        "SHOT_CLOCK": 10 + shot,
                        "PERIOD": 1,
                        "DRIBBLES": shot,
                        "TOUCH_TIME": 1.0,
                        "PTS_TYPE": 3 if shot == 3 else 2,
                        "FGM": int(shot % 2 == 1),
                        "PTS": 2 * int(shot % 2 == 1),
                        "W": "W",
                        "FINAL_MARGIN": 20,
                    }
                )
    return pd.DataFrame(rows)
