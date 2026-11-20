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

Only **02** is the pre-registered H2 decision. 01, 03 and 04 are exploratory
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
└── 04_brier_followup/
    ├── README.md
    ├── run_experiments.py        validation runner (cached Ollama calls)
    ├── analyze.py                metrics, recalibration, game bootstrap
    ├── make_figures.py
    └── results/                  predictions, summary.json, figures
```

Shared agent code (prompts, conditions, sampling) is in `src/nba_shots/agents.py`;
run every script from the repository root.

## Next

- **05:** anchored single-agent control on the same validation shots (fair H2 test once calibrated)
- **06:** the anchored conditions with a stronger hosted model
