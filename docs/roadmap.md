# Project Roadmap & Milestones

**Course**: CMSC473  
**Team**: Praharsh Nanduri, Kiran Manoj, Prannav Chettipalayam, Varun Kota

---

## Milestone Schedule

| Phase | Timeframe | Goals & Deliverables | Status | Lead / Owner |
|---|---|---|---|---|
| **Phase 1: Setup & Data Baselines** | Weeks 1–2 | - Ingest Kaggle 2014–15 shot logs<br>- Implement chronological whole-date splitting<br>- Implement career player history without temporal leakage<br>- Fit Constant, Logistic Regression, XGBoost baselines | **Complete** | Varun, Praharsh |
| **Phase 2: Vision Pipeline & Annotation** | Weeks 3–4 | - `nba_api` integration (`ShotChartDetail`, `VideoEventsAsset`)<br>- Video downloading and release frame decoding<br>- YOLO-Pose keypoint extraction with torso normalization<br>- Interactive web annotation tool (`nba-shots annotate`) | **Complete** | Prannav |
| **Phase 3: Multi-Agent System** | Weeks 5–6 | - Implement local Ollama client (`qwen3:4b-instruct`)<br>- CrewAI specialist agents (Context, History, Mechanics)<br>- Aggregator agent with calibrated probability synthesis<br>- Equal-call ablation controls (1-call vs 4-call budgets) | **Complete** | Kiran |
| **Phase 4: Video Annotation & Inference Run** | Weeks 7–8 | - Annotate 50+ downloaded clips using the annotation tool<br>- Run YOLO-Pose feature extraction on annotated clips<br>- Train `xgboost_cv` baseline on combined multimodal cohort<br>- Execute LLM inference under deterministic caching | **In Progress** | All Team |
| **Phase 5: Hypothesis Testing & Final Report** | Weeks 9–10 | - Paired game-clustered bootstrap analysis ($\Delta \text{Brier}$)<br>- Permutation feature importance analysis<br>- Failure case analysis (large prediction divergence)<br>- Final paper, slide deck, and project demo | **Upcoming** | All Team |

---

## Active Branches & Ownership

| Branch | Focus Area | Status | Primary Contributor |
|---|---|---|---|
| `main` | Production stable baseline pipeline | Up to date | Team |
| `feature/video-annotation-tool` | Interactive browser-based video annotation tool | Merged & pushed | Prannav |
| `feature/xgboost-importance` | Permutation Brier feature importance for XGBoost | Ready for review | Praharsh |
| `docs/prd-spec` | PRD, Technical Spec, Experiment Guide, and Roadmap | In progress | Team |
