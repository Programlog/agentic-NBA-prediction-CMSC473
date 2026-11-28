# Final Reproducibility Audit

**Project:** NBA Shot Prediction with Computer Vision and Agentic AI

**Audit date:** November 27, 2026

**Scope:** Local environment, dependency configuration, H1/H2 result artifacts, automated tests, and repository-wide code quality checks.

## 1. Purpose

This audit documents which parts of the project were successfully verified and which requirements remain unresolved for independent reproduction.

The goal is to evaluate the repository's reproducibility without retraining models, modifying frozen experimental results, or making changes to the H1/H2 evaluation pipelines.

## 2. Environment and Dependencies

The following environment was verified on Windows:

| Component | Result |
|---|---|
| Python | 3.11.4 |
| uv | 0.11.28 |
| pyproject.toml | Present |
| uv.lock | Present |
| README.md | Present |
| Proposal completion audit | Present |
| Dependency-lock check | Passed |

The command `uv lock --check` completed successfully and resolved 205 packages.

This confirms that the existing lock file is consistent with the current project configuration. It does not demonstrate that all dependencies can be installed successfully on a fresh machine.

## 3. Automated Test Verification

The following command was executed:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Results:

- 384 tests passed
- 2 tests skipped
- Runtime: approximately 17.5 seconds

The existing local test suite passed without failures.

However, the skipped tests were not independently investigated, and passing unit tests does not establish complete end-to-end experiment reproducibility.

## 4. H1 Artifact Verification

The following H1 paths were confirmed to exist locally:

- `docs/h1_results.md`
- `docs/h1_test_protocol.md`
- `reports/basketevent_h1_test/summary.json`
- `reports/basketevent_h1_test/test_predictions_seed_mean.csv`
- `reports/basketevent_h1_test/h1_effect_estimates.png`

The three checked H1 result artifacts were also confirmed to be tracked by Git.

These files preserve the recorded experimental results and supporting analysis. However, the frozen H1 evaluation was not rerun during this audit.

## 5. H2 Artifact Verification

The following H2 paths were confirmed to exist locally:

- `experiments/h2/02_final_test_100/README.md`
- `experiments/h2/02_final_test_100/protocol.md`
- `experiments/h2/02_final_test_100/report/combined_metrics.csv`
- `experiments/h2/02_final_test_100/report/predictions.csv`

Both checked H2 result artifacts were confirmed to be tracked by Git.

The frozen H2 inference experiment was not rerun during this audit.

## 6. Repository-Wide Code Quality

Ruff lint and formatting checks were performed across the following directories:

- `src`
- `tests`
- `tools`

### Lint findings

The repository-wide lint check identified existing issues, including:

- Lines exceeding the configured length limit
- Unsorted imports
- Unused imports

These findings were recorded rather than automatically corrected.

### Formatting findings

The Ruff formatting check reported:

- 15 files would be reformatted
- 57 files were already formatted

Therefore, the repository does not currently pass all repository-wide Ruff checks.

No existing source files were reformatted as part of this audit.

## 7. External Data and Software Requirements

The repository's documentation identifies several requirements for reproducing the complete experiments.

### H1: Computer Vision

Full H1 reproduction may require:

- BasketEvent 2025–26 source data
- Hugging Face authentication for data collection
- Video files and corresponding trajectory information
- Computer-vision dependencies and appropriate computing resources

Some generated files and raw datasets are intentionally excluded from Git.

### H2: Agentic AI

Full H2 reproduction may require:

- Kaggle 2014–15 NBA shot logs
- The specified local Ollama runtime and model
- The frozen H2 experimental runner and configuration
- Required locally generated intermediate inputs

The H2 documentation distinguishes the original frozen runner from subsequent agent implementations.

These requirements were identified from existing project documentation rather than verified through a fresh end-to-end execution.

## 8. Verification Boundaries

### Successfully verified

1. Local Python version and dependency configuration files.
2. Dependency-lock consistency.
3. Execution of the automated test suite.
4. Presence of the nine checked H1/H2 protocol and result paths.
5. Git tracking of five key frozen result artifacts.
6. Repository-wide lint and formatting status.

### Not independently verified

1. Installation in a completely fresh environment.
2. Downloading and preparing every original dataset.
3. End-to-end execution of the H1 computer-vision pipeline.
4. Reproduction of frozen H2 model inference.
5. Independent regeneration of the reported experimental metrics.
6. GPU availability and hardware-equivalent performance.
7. Complete computational-cost measurements.
8. All optional demonstration commands.

These unverified items should not be described as completed independent reproductions.

## 9. Research Interpretation

H1 and H2 were evaluated on different datasets and shot cohorts.

The H1 experiment found a small improvement from box-derived features, but the frozen held-out confidence interval included zero.

The frozen H2 experiment favored the multi-agent LLM over its budget-matched single-agent comparison. However, XGBoost performed better, and subsequent exploratory experiments qualified the original multi-agent advantage.

The separate H1 and H2 results do not establish how all evaluated models would rank on one identical dataset.

The release-detection follow-up audits also identified uncertainty involving provisional human release intervals, geometric conditions, and missing ball trajectories.

These limitations remain important when interpreting the predictive results.

## 10. Conclusion

The repository successfully passed 384 automated tests, maintained a consistent dependency lock, and contained the checked H1/H2 result artifacts in version control.

However, the repository-wide Ruff checks identified existing issues, and the complete experiments were not independently reproduced from raw datasets during this audit.

The evidence therefore supports local test verification and availability of committed experimental artifacts, but not a claim of complete end-to-end reproducibility.

No frozen H1/H2 experiment, prediction, model configuration, or detector threshold was modified during this audit.