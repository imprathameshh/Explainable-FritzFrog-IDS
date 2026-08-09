"""Phase 15-16 — train XGBoost (primary) and boosting comparison models.

Models (guide §19-20):
  - XGBoost   : primary high-performance gradient-boosting model
  - LightGBM  : efficient boosting for large tabular data
  - AdaBoost  : traditional boosting comparison

(Random Forest — the bagging comparison — is trained in 05_train_baselines.py.)

All models train on the same split/subsample as the baselines for a fair
comparison, use class-imbalance handling (scale_pos_weight / balanced base),
and are evaluated on the VALIDATION set only (never test). Per model we record
validation metrics, training time, inference time, and on-disk model size.

Run from the project root: python src/06_train_xgboost.py
"""
from pathlib import Path
from time import perf_counter
import json

import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Which split track: "stratified" (primary) or "dayheld" (robustness).
SPLIT = "stratified"

DATA_DIR = Path("data/processed") / SPLIT
MODELS_DIR = Path("models")
MODEL_OUT_DIR = MODELS_DIR / "boosting" / SPLIT
RESULTS_DIR = Path("results/metrics")

RANDOM_STATE = 42
# Stratified training subsample (same convention as baselines); None = full.
TRAIN_SAMPLE = 2_000_000


def load_split(name: str, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_parquet(DATA_DIR / f"X_{name}.parquet")[feature_names]
    y = pd.read_parquet(DATA_DIR / f"y_{name}.parquet")["Target"]
    return X, y


def maybe_subsample(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if TRAIN_SAMPLE is None or len(X) <= TRAIN_SAMPLE:
        return X, y
    from sklearn.model_selection import train_test_split

    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=TRAIN_SAMPLE, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"  [sampled] {len(X_s):,}-row stratified subsample of {len(X):,} "
        f"(TRAIN_SAMPLE={TRAIN_SAMPLE:,})"
    )
    return X_s, y_s


def evaluate(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    t0 = perf_counter()
    predictions = model.predict(X_val)
    inference_time = perf_counter() - t0
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
        "inference_time_s": round(inference_time, 3),
    }


def main() -> None:
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    print(f"Split: {SPLIT} | Features: {len(feature_names)}")

    X_train_full, y_train_full = load_split("train", feature_names)
    X_val, y_val = load_split("validation", feature_names)
    print(f"Train: {X_train_full.shape} | Validation: {X_val.shape}")

    X_train, y_train = maybe_subsample(X_train_full, y_train_full)
    del X_train_full, y_train_full

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive
    print(f"scale_pos_weight = {scale_pos_weight:.3f}\n")

    models = {
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=3, class_weight="balanced", random_state=RANDOM_STATE
            ),
            n_estimators=200,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        t0 = perf_counter()
        if name == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif name == "lightgbm":
            import lightgbm as lgb

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)
        train_time = perf_counter() - t0

        metrics = evaluate(model, X_val, y_val)
        metrics["train_time_s"] = round(train_time, 3)
        metrics["train_rows"] = int(len(X_train))

        model_path = MODEL_OUT_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        metrics["model_size_mb"] = round(model_path.stat().st_size / 1e6, 3)
        results[name] = metrics

        print(
            f"  f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} "
            f"precision={metrics['precision']:.4f} pr_auc={metrics['pr_auc']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} (train {train_time:.1f}s)\n"
        )

    with open(RESULTS_DIR / f"model_metrics_{SPLIT}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table = pd.DataFrame(
        {
            name: {
                k: m[k]
                for k in (
                    "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "false_positive_rate", "train_time_s", "inference_time_s",
                    "model_size_mb", "train_rows",
                )
            }
            for name, m in results.items()
        }
    ).T
    table.to_csv(RESULTS_DIR / f"model_comparison_{SPLIT}.csv")

    print(f"=== Validation comparison [{SPLIT}] (sorted by PR-AUC) ===")
    print(table.sort_values("pr_auc", ascending=False).to_string())
    print("\nSaved models to", MODEL_OUT_DIR)


if __name__ == "__main__":
    main()
