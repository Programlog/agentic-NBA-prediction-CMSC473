# Experiment 06 test protocol: shooter history

**Status:** frozen on 2026-11-20 after the validation runs and the contamination
probe, and **before any test-split call** in this experiment. This experiment is
exploratory. The pre-registered H2 answer is still [02](../02_final_test_100/README.md).

## Question

The LLM agents add nothing beyond the 9 shared tabular features (experiment 05).
Does giving them **shooter history** help? This means two things:

- **Arm A:** the player's earlier make rates.
- **Arm B:** the same rates plus the player's **name**, so the model can use its own knowledge.

## Data and features

| Setting | Frozen value |
|---|---|
| Base data | `artifacts/day1/prepared/shots.parquet` (`4528a5be…`) |
| History data | `artifacts/h2_history/shots.parquet` (`e40b0edf…`), built by `prepare.py` (`f2b01e46…`) |
| History features | `player_prior_attempts` and `player_prior_make_rate` (existing `add_player_history`, Beta(1,1)); `player_band_attempts` and `player_band_rate` (the same distance band as 04, shrunk toward the train league band rate with k = 20). All use **strictly earlier dates only** |
| Leakage check | 20 random selected shots were recomputed from earlier-date shots: 0 mismatches |
| Baselines with history | constant, LR and XGBoost retrained on the 9 base features + 4 history features, with day-1 preprocessing and hyperparameters. The same local pipeline without history reproduces day-1 XGBoost (max difference 1e-16) |
| Shots | the same 100 validation and 100 test shots as 02, 04 and 05 (checked) |

## Conditions and model

GPT-6 Luna (`gpt-6-luna`), reasoning effort `none`, temperature 0, seed 42.
Every condition uses the 04 anchors and a budget of 4 calls per shot.

| Condition | Description |
|---|---|
| `multi_hist` | 3 specialists (the Player history analyst is now on) + aggregator; history lines go to the history analyst and the aggregator |
| `single_hist` | one analyst, same history lines, 4 sequential calls |
| `multi_named` / `single_named` | as above, plus "The shooter is {name}…" |

No-history comparison: 05's Luna `none` `multi_anchored` and `single_budget_anchored`
predictions on the same shots.

Code: `run_experiments.py` `f9907b76…`, `analyze.py` `f8f695b7…`,
`contamination.py` `21486f1a…`; `src/nba_shots/agents.py` `544486e3…` (unchanged).

## Validation (used only to design this protocol)

Brier on 100 shots (44% made):

| | no history (05) | history | history + name |
|---|---:|---:|---:|
| multi | 0.2392 | 0.2386 | 0.2387 |
| single | 0.2417 | 0.2405 | 0.2394 |

XGBoost is 0.2318 (0.2341 with history). Every validation difference had an
interval that included 0.

**Contamination probe** (135 players, with no shot context): Luna's 2014–15 FG%
correlated 0.917 with the FG% in this data. Its mean error was 1.2 points, it was
within 1 point for 65% of players, and it said "high" confidence every time. So the
model clearly **remembers season-level shooting stats**. Arm B's player knowledge
therefore includes season outcomes. That knowledge is coarse, a season average,
but it does partly cover each test shot's own season. The earlier-dates history
rate already in the prompts tracks the same quantity even more closely (correlation
0.973, mean error 0.9 points), so arm B carries little information beyond arm A.

## Analysis (fixed now)

- **Metric:** Brier. Paired 95% intervals come from 1,000 game-bootstrap resamples, seed 42.
- **Primary comparisons** (test):
  1. `hist/multi − nohist/multi`: does history data help the agents?
  2. `named/multi − hist/multi`: does knowing the player help?
- **Secondary comparisons:**
  - the same two comparisons for single
  - multi − single within each arm
  - each LLM arm − XGBoost with history
  - XGBoost with history − XGBoost
  - LR with history − LR
- An interval entirely below 0 favours the first term. One that contains 0 means
  no detectable difference. There is no multiplicity correction.

## Execution

```bash
uv run --frozen --all-extras python experiments/h2/06_shooter_history/run_experiments.py --split test
uv run --frozen --all-extras python experiments/h2/06_shooter_history/analyze.py
```

Acceptance: 100/100 `ok` predictions for every condition. Spend cap: $5. The
validation run cost $0.23 for 1,400 fresh calls; the identical context and
mechanics specialist calls are shared between the two arms through the cache.
