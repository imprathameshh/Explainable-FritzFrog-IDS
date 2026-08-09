"""Phase 20-21 — feature selection and full-vs-reduced comparison.

Two-stage selection (guide §24, the practical approach for large IDS data):
  1. Remove constant (zero-variance) features and highly-correlated redundant
     features.
  2. Rank the rest by SHAP importance and evaluate reduced models (top 10/20/
     30/40 features) against the full model.

Also runs a DST-PORT ABLATION: SHAP showed Dst Port dominates, which is a likely
testbed shortcut (attacks use fixed ports). Retraining without it measures how
much detection depends on that artifact.

Every model is XGBoost (depth-6), trained on the stratified train subsample and
evaluated on validation (never test). Records metrics, feature count, train/
inference time, and model size — the efficiency contribution.

Run from the project root: python src/09_feature_selection.py
"""
from pathlib import Path
from time import perf_counter
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier

SPLIT = "stratified"
DATA_DIR = Path("data/processed") / SPLIT
MODELS_DIR = Path("models")
MODEL_OUT_DIR = MODELS_DIR / "feature_selection" / SPLIT
RESULTS_DIR = Path("results/metrics")

RANDOM_STATE = 42
TRAIN_SAMPLE = 2_000_000
CORR_SAMPLE = 100_000
CORR_THRESHOLD = 0.95
TOP_K = [10, 20, 30, 40]


def load_split(name, feats):
    X = pd.read_parquet(DATA_DIR / f"X_{name}.parquet")[feats]
    y = pd.read_parquet(DATA_DIR / f"y_{name}.parquet")["Target"]
    return X, y


def subsample(X, y, n):
    if n is None or len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=n, random_state=RANDOM_STATE, stratify=y
    )
    return Xs, ys


def make_model(scale_pos_weight):
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=scale_pos_weight, early_stopping_rounds=30,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def evaluate(model, X_val, y_val):
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
    }


def correlated_to_drop(X_sample, importance):
    """Drop the lower-importance member of each |corr|>threshold pair."""
    corr = X_sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = set()
    for col in upper.columns:
        for row in upper.index[upper[col] > CORR_THRESHOLD]:
            if row in drop or col in drop:
                continue
            # keep the more important one
            drop.add(col if importance.get(row, 0) >= importance.get(col, 0) else row)
    return sorted(drop)


def main() -> None:
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    importance = pd.read_csv(
        RESULTS_DIR / "shap_feature_importance.csv", index_col=0
    )["mean_abs_shap"]

    X_train_full, y_train_full = load_split("train", feature_names)
    X_val, y_val = load_split("validation", feature_names)
    scale_pos_weight = (y_train_full == 0).sum() / (y_train_full == 1).sum()
    X_train, y_train = subsample(X_train_full, y_train_full, TRAIN_SAMPLE)
    del X_train_full, y_train_full

    # Stage 1: constants + correlated redundancy.
    variances = X_train.var()
    constant = variances[variances == 0].index.tolist()
    corr_sample = X_train.sample(n=min(CORR_SAMPLE, len(X_train)), random_state=RANDOM_STATE)
    correlated = correlated_to_drop(corr_sample.drop(columns=constant), importance)
    reduced_stage1 = [f for f in feature_names if f not in set(constant) | set(correlated)]
    ranked = [f for f in importance.index if f in reduced_stage1]
    print(f"constant dropped: {len(constant)} | correlated dropped: {len(correlated)} "
          f"| remaining: {len(reduced_stage1)}")

    experiments = {
        "full": feature_names,
        "no_constant": [f for f in feature_names if f not in set(constant)],
        "no_dst_port": [f for f in feature_names if f != "Dst Port"],
        "stage1_reduced": reduced_stage1,
    }
    for k in TOP_K:
        experiments[f"top{k}_shap"] = importance.head(k).index.tolist()

    results = {}
    for name, feats in experiments.items():
        feats = [f for f in feats if f in feature_names]
        model = make_model(scale_pos_weight)
        t0 = perf_counter()
        model.fit(X_train[feats], y_train,
                  eval_set=[(X_val[feats], y_val)], verbose=False)
        train_time = perf_counter() - t0
        t1 = perf_counter()
        metrics = evaluate(model, X_val[feats], y_val)
        infer_time = perf_counter() - t1
        path = MODEL_OUT_DIR / f"xgboost_{name}.joblib"
        joblib.dump(model, path)
        metrics.update({
            "n_features": len(feats),
            "train_time_s": round(train_time, 2),
            "inference_time_s": round(infer_time, 3),
            "model_size_mb": round(path.stat().st_size / 1e6, 3),
        })
        results[name] = metrics
        print(f"  {name:16s} n={len(feats):>2} f1={metrics['f1']:.4f} "
              f"pr_auc={metrics['pr_auc']:.4f} fpr={metrics['false_positive_rate']:.4f} "
              f"({train_time:.0f}s)")

    with open(RESULTS_DIR / f"feature_selection_{SPLIT}.json", "w", encoding="utf-8") as f:
        json.dump({"constant": constant, "correlated": correlated,
                   "stage1_reduced": reduced_stage1, "results": results}, f, indent=2)
    with open(RESULTS_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"stage1_reduced": reduced_stage1,
                   "top20_shap": importance.head(20).index.tolist()}, f, indent=2)

    table = pd.DataFrame(results).T[
        ["n_features", "precision", "recall", "f1", "roc_auc", "pr_auc",
         "false_positive_rate", "train_time_s", "inference_time_s", "model_size_mb"]
    ]
    table.to_csv(RESULTS_DIR / f"feature_selection_comparison_{SPLIT}.csv")
    print("\n=== Full vs reduced (validation) ===")
    print(table.to_string())
    # Dst Port ablation callout
    full_pr = results["full"]["pr_auc"]
    nodp_pr = results["no_dst_port"]["pr_auc"]
    print(f"\nDst Port ablation: PR-AUC {full_pr:.4f} -> {nodp_pr:.4f} "
          f"(delta {nodp_pr - full_pr:+.4f})")


if __name__ == "__main__":
    main()
