"""Phase 18 — hyperparameter tuning for the primary model (XGBoost).

RandomizedSearchCV over train/validation ONLY — the test set is never touched
(guide §22). Scoring is average_precision (PR-AUC), consistent with the
class-imbalance focus. To keep cross-validation tractable at this scale, the
search runs on a stratified TUNE_SAMPLE subsample; the best configuration is
then refit on the larger TRAIN_SAMPLE with early stopping and compared against
the untuned XGBoost from 06 on the validation set.

Run from the project root: python src/06b_tune_xgboost.py
"""
from pathlib import Path
from time import perf_counter
import json

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

SPLIT = "stratified"
DATA_DIR = Path("data/processed") / SPLIT
MODELS_DIR = Path("models")
MODEL_OUT_DIR = MODELS_DIR / "boosting" / SPLIT
RESULTS_DIR = Path("results/metrics")

RANDOM_STATE = 42
TUNE_SAMPLE = 500_000    # rows used inside cross-validated search
TRAIN_SAMPLE = 2_000_000  # rows used to refit the best config (matches 06)
N_ITER = 25
CV_FOLDS = 3

PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 400, 600],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0],
}


def load_split(name: str, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_parquet(DATA_DIR / f"X_{name}.parquet")[feature_names]
    y = pd.read_parquet(DATA_DIR / f"y_{name}.parquet")["Target"]
    return X, y


def subsample(X, y, n):
    if n is None or len(X) <= n:
        return X, y
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=n, random_state=RANDOM_STATE, stratify=y
    )
    return X_s, y_s


def evaluate(model, X_val, y_val) -> dict:
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "pr_auc": float(average_precision_score(y_val, probabilities)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    X_train_full, y_train_full = load_split("train", feature_names)
    X_val, y_val = load_split("validation", feature_names)

    negative = int((y_train_full == 0).sum())
    positive = int((y_train_full == 1).sum())
    scale_pos_weight = negative / positive
    print(f"Split: {SPLIT} | scale_pos_weight={scale_pos_weight:.3f}")

    # --- search on a subsample ---
    X_tune, y_tune = subsample(X_train_full, y_train_full, TUNE_SAMPLE)
    print(f"Tuning on {len(X_tune):,} rows, {N_ITER} candidates x {CV_FOLDS}-fold CV")

    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=1,  # XGBoost already uses all cores per fit; avoid oversubscription
        verbose=2,
    )
    t0 = perf_counter()
    search.fit(X_tune, y_tune)
    search_time = perf_counter() - t0
    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    print(f"Search time: {search_time:.1f}s\n")

    # --- refit best config on the larger sample with early stopping ---
    X_train, y_train = subsample(X_train_full, y_train_full, TRAIN_SAMPLE)
    del X_train_full, y_train_full

    best = dict(search.best_params_)
    tuned = XGBClassifier(
        **best,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = perf_counter()
    tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    refit_time = perf_counter() - t0
    tuned_metrics = evaluate(tuned, X_val, y_val)

    joblib.dump(tuned, MODEL_OUT_DIR / "xgboost_tuned.joblib")

    # --- compare against untuned XGBoost from 06 ---
    untuned_metrics = None
    untuned_path = MODEL_OUT_DIR / "xgboost.joblib"
    if untuned_path.exists():
        untuned_metrics = evaluate(joblib.load(untuned_path), X_val, y_val)

    out = {
        "best_params": search.best_params_,
        "cv_pr_auc": float(search.best_score_),
        "search_time_s": round(search_time, 1),
        "refit_time_s": round(refit_time, 1),
        "tune_rows": int(len(X_tune)),
        "refit_rows": int(len(X_train)),
        "validation_tuned": tuned_metrics,
        "validation_untuned": untuned_metrics,
    }
    with open(RESULTS_DIR / f"xgboost_tuning_{SPLIT}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== Tuned vs untuned XGBoost (validation) ===")
    for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate"):
        t = tuned_metrics[metric]
        u = untuned_metrics[metric] if untuned_metrics else float("nan")
        print(f"  {metric:20s} tuned={t:.4f}  untuned={u:.4f}  delta={t - u:+.4f}")
    print(f"\nSaved tuned model to {MODEL_OUT_DIR/'xgboost_tuned.joblib'}")
    print(f"Saved xgboost_tuning_{SPLIT}.json")


if __name__ == "__main__":
    main()
