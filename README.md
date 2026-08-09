# Hybrid-ML-FritzFrog

MSc thesis practical work: an **explainable machine-learning framework for detecting
FritzFrog-like malicious network behaviour** using the CSE-CIC-IDS2018 dataset.

> **Research limitation.** CSE-CIC-IDS2018 does **not** contain confirmed FritzFrog
> traffic. This project studies FritzFrog-*like* behaviour (SSH brute-force, botnet,
> DoS/DDoS, anomalous flows) as a proxy — it is **not** a detector trained on authentic
> FritzFrog malware. See `THESIS_OVERVIEW.md`.

## Results at a glance

Primary model: **XGBoost** (depth-6), binary benign-vs-attack, on the 15.7M-flow dataset.

| Evaluation | F1 | Recall | Precision | ROC-AUC | PR-AUC | FPR |
|---|--:|--:|--:|--:|--:|--:|
| **Stratified test** (headline, 2.36M flows) | 0.957 | 0.950 | 0.965 | 0.990 | 0.978 | 0.61% |
| **Day-held test** (unseen day, Web attacks, 566) | 0.548 | 0.486 | 0.629 | 0.825 | 0.523 | 0.02% |

Key findings:
- **Boosting beats bagging**: XGBoost/LightGBM PR-AUC 0.978/0.987 > Random Forest 0.970 > baselines.
- **10 SHAP-ranked features ≈ all 78** (PR-AUC 0.975 vs 0.978) — a large efficiency gain.
- **`Dst Port` dominates SHAP but is not load-bearing** (ablation −0.0011 PR-AUC) — the testbed-port shortcut isn't critical.
- **Infiltration and Web attacks are the hard families** (day-held generalisation is where scores drop); all others detect near-perfectly.

## Project structure

```
data/{raw,interim,processed}/       raw CSVs -> cleaned -> combined (+ stratified/ & dayheld/ splits); gitignored
src/
  00_download_data.sh               download CSE-CIC-IDS2018 CSVs (precondition)
  01_validate_data.py               gate: raw non-empty, readable, Label present
  02_preprocess.py + cleaning.py    per-file cleaning -> data/interim/
  03_combine_data.py                schema-align + merge -> combined_cleaned.parquet
  04_prepare_splits.py              stratified (primary) + attack-preserving day-based (secondary) splits
  05_train_baselines.py             Dummy, LogReg, Decision Tree, Random Forest
  06_train_xgboost.py               XGBoost (primary) + LightGBM + AdaBoost
  06b_tune_xgboost.py               RandomizedSearchCV (train/val only)
  08_shap_analysis.py               SHAP: beeswarm, bar, waterfalls, dependence
  09_feature_selection.py           two-stage selection + SHAP top-K + Dst Port ablation
  10_train_final_model.py           final XGBoost, scored ONCE on test (both tracks)
  11_generate_figures.py            dissertation figures
  12_demo_prediction.py             optional CLI demo (score a flow CSV + SHAP drivers)
tests/                              pytest unit tests for the cleaning logic
notebooks/02_eda.ipynb              exploratory data analysis
models/  results/{metrics,predictions}/  figures/   outputs
docs/superpowers/specs/             design spec
THESIS_OVERVIEW.md                  problem, method, progress, threats to validity
```

## Setup

Requires Python 3.10+, and (for XGBoost/LightGBM on macOS) the OpenMP runtime:

```bash
brew install libomp                 # macOS only; XGBoost/LightGBM native dependency
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce (run order)

```bash
# 0. Prerequisite: download the dataset (needs AWS CLI; no account required)
bash src/00_download_data.sh

# 1. Data pipeline
python src/01_validate_data.py
python src/02_preprocess.py
python src/03_combine_data.py
python src/04_prepare_splits.py         # writes data/processed/{stratified,dayheld}/

# 2. EDA (optional, interactive)
jupyter lab notebooks/02_eda.ipynb

# 3. Modelling (default SPLIT="stratified" inside each script)
python src/05_train_baselines.py
python src/06_train_xgboost.py
python src/06b_tune_xgboost.py
python src/08_shap_analysis.py
python src/09_feature_selection.py
python src/10_train_final_model.py      # both tracks; scores test once
python src/11_generate_figures.py

# 4. Optional demo
python src/12_demo_prediction.py path/to/flows.csv --explain
```

## Tests

```bash
pytest                                  # cleaning-logic unit tests
```

## Notes on scale & reproducibility

- The largest raw CSV is 3.8 GB (~7.9M rows); the combined dataset is 15.7M × 80.
- Modelling scripts use a stratified `TRAIN_SAMPLE` (default 2M) for tractability, logged in output.
- The **test set is evaluated only in `10`** (leakage control). Tuning/selection use validation only.
- Fixed `random_state=42` throughout. Metrics/figures are regenerable outputs (not committed).

## Two evaluation tracks

- **Stratified** (primary): representative headline metrics, all 15 classes in every split.
- **Day-held** (secondary): attack-preserving day-based holdout — a cross-day generalisation
  experiment (test = held-out Web-attack day). See `THESIS_OVERVIEW.md` §6 for the rationale
  and the base-rate caveat (the day-held test is 99.95% benign).
