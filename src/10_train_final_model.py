"""Phase 22 — final model, evaluated ONCE on the untouched test set.

For each evaluation track (stratified primary, day-held robustness):
  1. combine train + validation, retrain the final XGBoost (depth-6, full
     features — the parsimonious choice; tuning gave no real gain),
  2. evaluate a single time on the test set,
  3. save the model, test predictions/probabilities, and metrics.

This is the only place the test set is touched (guide §26). The day-held test is
99.95% benign (566 attacks), so its precision is base-rate limited — reported
with PR-AUC / recall / FPR, not accuracy.

Run from the project root: python src/10_train_final_model.py
"""
from pathlib import Path
from time import perf_counter
import json

import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier

TRACKS = ["stratified", "dayheld"]
PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/metrics")
PRED_DIR = Path("results/predictions")

RANDOM_STATE = 42
# Rows used to fit the final model. None = FULL train+val (streamed as float32
# to bound memory); an int = stratified subsample.
FINAL_SAMPLE = None
BATCH_ROWS = 1_000_000


def load(track, name, feats):
    d = PROCESSED / track
    X = pd.read_parquet(d / f"X_{name}.parquet")[feats]
    y = pd.read_parquet(d / f"y_{name}.parquet")["Target"]
    return X, y


def subsample(X, y, n):
    if n is None or len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=n, random_state=RANDOM_STATE, stratify=y
    )
    return Xs, ys


def load_full_fit(track: str, feats: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Stream full train+val as float32 to bound peak memory."""
    d = PROCESSED / track
    X_parts, y_parts = [], []
    for split in ("train", "validation"):
        y_parts.append(pd.read_parquet(d / f"y_{split}.parquet")["Target"])
        reader = pq.ParquetFile(d / f"X_{split}.parquet")
        for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=feats):
            X_parts.append(batch.to_pandas().astype("float32"))
    X = pd.concat(X_parts, ignore_index=True)
    del X_parts
    y = pd.concat(y_parts, ignore_index=True)
    return X, y


def run_track(track: str, feats: list[str]) -> dict:
    print(f"\n=== {track} ===")
    if FINAL_SAMPLE is None:
        X_fit, y_fit = load_full_fit(track, feats)
        print(f"  fit rows: {len(X_fit):,} (FULL train+val, float32) | "
              f"attack fraction {y_fit.mean():.4f}")
    else:
        len_tr = len(pd.read_parquet(PROCESSED / track / "y_train.parquet"))
        len_va = len(pd.read_parquet(PROCESSED / track / "y_validation.parquet"))
        tr_n = int(FINAL_SAMPLE * len_tr / (len_tr + len_va))
        va_n = FINAL_SAMPLE - tr_n
        # Sample each split before concat to bound peak memory.
        X_tr, y_tr = load(track, "train", feats)
        X_tr, y_tr = subsample(X_tr, y_tr, tr_n)
        X_va, y_va = load(track, "validation", feats)
        X_va, y_va = subsample(X_va, y_va, va_n)
        X_fit = pd.concat([X_tr, X_va], ignore_index=True)
        y_fit = pd.concat([y_tr, y_va], ignore_index=True)
        del X_tr, X_va, y_tr, y_va
        print(f"  fit rows: {len(X_fit):,} | attack fraction {y_fit.mean():.4f}")

    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1,
    )
    t0 = perf_counter()
    model.fit(X_fit, y_fit)
    train_time = perf_counter() - t0
    del X_fit, y_fit

    X_test, y_test = load(track, "test", feats)
    t1 = perf_counter()
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    infer_time = perf_counter() - t1

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_rows": int(len(y_test)),
        "test_attack": int(y_test.sum()),
        "train_time_s": round(train_time, 2),
        "inference_time_s": round(infer_time, 3),
    }

    out_dir = MODELS_DIR / "final" / track
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "xgboost_final.joblib")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": pred, "proba": proba}).to_parquet(
        PRED_DIR / f"final_test_predictions_{track}.parquet", index=False
    )
    print(f"  test: f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} "
          f"precision={metrics['precision']:.4f} pr_auc={metrics['pr_auc']:.4f} "
          f"roc_auc={metrics['roc_auc']:.4f} fpr={metrics['false_positive_rate']:.4f}")
    return metrics


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feats = joblib.load(MODELS_DIR / "feature_names.joblib")
    all_metrics = {track: run_track(track, feats) for track in TRACKS}
    with open(RESULTS_DIR / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nSaved final_test_metrics.json and per-track predictions/models.")


if __name__ == "__main__":
    main()
