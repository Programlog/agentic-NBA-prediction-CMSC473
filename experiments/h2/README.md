# H2 Experiments: Multi-Agent vs. Single-Agent LLMs

**H2:** when given the same features, does a multi-agent LLM pipeline predict NBA
shot success better than a single LLM and conventional models such as XGBoost?

Every H2 experiment lives in this folder, in the order it was run. Each
subfolder holds its write-up (`README.md`), the scripts that produced it, and
its committed results. Raw run caches stay in `artifacts/`, which Git ignores.

| # | Experiment | Split | Status | Headline result |
| --- | --- | --- | --- | --- |
| 01 | [30-shot validation pilot](01_validation_pilot_30/README.md) | validation, 30 shots | complete | No multi-agent benefit on 30 shots; used to develop the frozen protocol |
| 02 | [Final held-out test](02_final_test_100/README.md) (**pre-registered H2 result**) | test, 100 shots | complete, frozen | Multi-agent beat the budget-matched single LLM: Brier −0.025 [−0.044, −0.007]. No LLM beat XGBoost (0.206) |
| 03 | [Disagreement-aware aggregation](03_disagreement_aggregation/pilot_4_shots.md) | validation, 4 and 7 shots | exploratory, partial | Small Brier gains on very few shots; inconclusive |
| 04 | [Fixing LLM overconfidence](04_brier_followup/README.md) | validation, 100 shots (+ test recalibration) | complete, exploratory | Anchoring to training make rates: Brier 0.273 → 0.238. Recalibrated multi- and single-agent tie on test |
| 05 | [Stronger model: GPT-6 Luna](05_stronger_model/README.md) | validation + test, 100 shots each | complete, exploratory | Luna beats qwen by 0.03–0.06 Brier unanchored, but anchored qwen matches it (0.218 vs. 0.217); multi- and single-agent tie (−0.0004 [−0.002, +0.001]); best LLM 0.213 vs. XGBoost 0.206 |
| 06 | [Shooter history](06_shooter_history/README.md) | validation + test, 100 shots each | complete, exploratory | History helps multi-agent slightly (−0.004 [−0.009, −0.0003]); multi then beats single (−0.004 [−0.008, −0.001]); player names add nothing (model memorized season FG%); XGBoost still best |
| 07 | [Better prompts, similar shots, XGBoost as a tool](07_prompt_tuning/README.md) | validation 500 + test 1,000 shots | complete, exploratory | Rewritten prompts and similar shots don't reach XGBoost; with the XGBoost estimate as a tool the multi-agent system **ties** it (+0.0001 [−0.002, +0.002]) by mostly copying it; multi = single again |

Only **02** is the pre-registered H2 decision. 01 and 03–07 are exploratory
and do not change it.

## Layout

```
experiments/h2/
├── 01_validation_pilot_30/       write-up only
├── 02_final_test_100/
│   ├── protocol.md               frozen before test inference
│   ├── README.md                 results, limitations, reproduction
│   ├── run_llm_final.sh          frozen runner (reproduce at commit 64ee854)
│   ├── make_figures.py
│   └── report/                   metrics, paired comparisons, predictions, figures
├── 03_disagreement_aggregation/
│   ├── pilot_4_shots.md
│   └── validation_10_partial.md
├── 04_brier_followup/
│   ├── README.md
│   ├── run_experiments.py        validation runner (cached Ollama calls)
│   ├── analyze.py                metrics, recalibration, game bootstrap
│   ├── make_figures.py
│   └── results/                  predictions, summary.json, figures
├── 05_stronger_model/
│   ├── README.md
│   ├── protocol.md               frozen before test calls
│   ├── openai_client.py          OpenAI client (cost cap, throttle)
│   ├── run_experiments.py        Ollama or OpenAI, validation or test
│   ├── analyze.py
│   ├── make_figures.py
│   └── results/                  per-run predictions, summary.json, figures
├── 06_shooter_history/
│   ├── README.md
│   ├── protocol.md               frozen before test calls
│   ├── prepare.py                history features + baselines with history
│   ├── run_experiments.py
│   ├── contamination.py          does the model remember season FG%?
│   ├── analyze.py
│   ├── make_figures.py
│   └── results/
└── 07_prompt_tuning/
    ├── README.md
    ├── protocol.md               validation rules + test protocol, frozen before calls
    ├── prepare.py                selections, kNN evidence, rate tables, baselines
    ├── prompts.py                prompt variants V0–V4
    ├── run_experiments.py
    ├── analyze.py
    ├── make_figures.py
    └── results/
```

Shared agent code (prompts, conditions, sampling) is in `src/nba_shots/agents.py`;
run every script from the repository root.

## Next

- `gpt-5.6-luna` as a third model once the API project has access
- Replicate 06's small multi-agent gain on a larger test sample before relying on it
- 07 suggests the tabular features are the ceiling: beating XGBoost needs new
  information (H1 video features, shot type, play context), not better prompts
