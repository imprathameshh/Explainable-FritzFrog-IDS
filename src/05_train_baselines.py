"""Phase 14 — train baseline models and evaluate on the VALIDATION set.

Baselines establish the bar XGBoost (06) must beat (guide §18):
  - Dummy (most_frequent)  -> proves the model beats guessing the majority class
  - Logistic Regression    -> linear baseline (scaled, inside a Pipeline)
  - Decision Tree          -> single-tree baseline
  - Random Forest          -> bagging ensemble baseline

All models are selected/compared on the VALIDATION set, never the test set
(leakage control). Class imbalance is handled with class_weight="balanced".

Scale note: at ~14.4M training rows, Random Forest with n_jobs=-1 duplicates the
data per worker and can exhaust memory. A stratified training subsample
(TRAIN_SAMPLE) is applied and LOGGED (not silent); set it to None for full data.

Run from the project root: python src/05_train_baselines.py
"""
from pathlib import Path
from time import perf_counter
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Which split track to train on: "stratified" (primary) or "dayheld" (robustness).
SPLIT = "stratified"

DATA_DIR = Path("data/processed") / SPLIT
MODELS_DIR = Path("models")
MODEL_OUT_DIR = MODELS_DIR / "baselines" / SPLIT
RESULTS_DIR = Path("results/metrics")

RANDOM_STATE = 42
# Stratified training subsample for tractability. None = full training set.
# Applied to every model uniformly and logged. Tune to your machine's RAM.
TRAIN_SAMPLE = 2_000_000
RF_ESTIMATORS = 200


def load_split(name: str, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_parquet(DATA_DIR / f"X_{name}.parquet")[feature_names]
    y = pd.read_parquet(DATA_DIR / f"y_{name}.parquet")["Target"]
    return X, y


def maybe_subsample(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if TRAIN_SAMPLE is None or len(X) <= TRAIN_SAMPLE:
        return X, y
    # Stratified subsample to preserve the class ratio.
    from sklearn.model_selection import train_test_split

    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=TRAIN_SAMPLE, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"  [sampled] training on a stratified {len(X_s):,}-row subsample "
        f"of {len(X):,} (TRAIN_SAMPLE={TRAIN_SAMPLE:,})"
    )
    return X_s, y_s


def evaluate(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    t0 = perf_counter()
    predictions = model.predict(X_val)
    inference_time = perf_counter() - t0

    # Probabilities for ROC-AUC / PR-AUC where available.
    try:
        probabilities = model.predict_proba(X_val)[:, 1]
        roc_auc = float(roc_auc_score(y_val, probabilities))
        pr_auc = float(average_precision_score(y_val, probabilities))
    except (AttributeError, ValueError):
        roc_auc = float("nan")
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": float(fpr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "inference_time_s": round(inference_time, 3),
    }


def build_models() -> dict:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def main() -> None:
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    print(f"Split: {SPLIT} | Features: {len(feature_names)}")

    X_train_full, y_train_full = load_split("train", feature_names)
    X_val, y_val = load_split("validation", feature_names)
    print(f"Train: {X_train_full.shape} | Validation: {X_val.shape}")
    print(f"Validation attack fraction: {y_val.mean():.4f}")

    # Sample once so every model trains on the same rows and the full frame
    # is freed before any model runs (keeps peak memory bounded).
    X_train, y_train = maybe_subsample(X_train_full, y_train_full)
    del X_train_full, y_train_full
    print(f"Training rows used: {len(X_train):,}\n")

    results = {}
    for name, model in build_models().items():
        print(f"Training {name} ...")
        t0 = perf_counter()
        model.fit(X_train, y_train)
        train_time = perf_counter() - t0

        metrics = evaluate(model, X_val, y_val)
        metrics["train_time_s"] = round(train_time, 3)
        metrics["train_rows"] = int(len(X_train))
        results[name] = metrics

        joblib.dump(model, MODEL_OUT_DIR / f"{name}.joblib")
        print(
            f"  f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} "
            f"precision={metrics['precision']:.4f} pr_auc={metrics['pr_auc']:.4f} "
            f"(train {train_time:.1f}s)\n"
        )

    with open(RESULTS_DIR / f"baseline_metrics_{SPLIT}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Flat comparison table for the thesis.
    table = pd.DataFrame(
        {
            name: {
                k: m[k]
                for k in (
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                    "false_positive_rate",
                    "train_time_s",
                    "inference_time_s",
                    "train_rows",
                )
            }
            for name, m in results.items()
        }
    ).T
    table.to_csv(RESULTS_DIR / f"baseline_comparison_{SPLIT}.csv")

    print(f"=== Validation comparison [{SPLIT}] (sorted by PR-AUC) ===")
    print(table.sort_values("pr_auc", ascending=False).to_string())
    print("\nSaved models to", MODEL_OUT_DIR)
    print(f"Saved baseline_metrics_{SPLIT}.json and baseline_comparison_{SPLIT}.csv to", RESULTS_DIR)


if __name__ == "__main__":
    main()
