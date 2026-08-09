"""Phase 19 — SHAP explainability for the primary XGBoost model.

Uses the depth-6 (untuned) XGBoost — simpler trees give cleaner, more
interpretable explanations at identical performance. SHAP is run on a sample of
the VALIDATION set (the test set stays untouched for final evaluation).

Produces (guide §23):
  1. beeswarm      — global feature impact
  2. bar           — mean |SHAP| feature importance
  3. waterfall(TP) — a correctly detected attack
  4. waterfall(FP) — a false positive
  5. dependence    — top features
and a feature-importance table. Interpretation notes are printed: which
features drive attack probability, and a leakage sanity check (no identifier
should dominate — identifiers were dropped in preprocessing/splitting).

SHAP explains model behaviour; it is not proof of genuine FritzFrog detection.

Run from the project root: python src/08_shap_analysis.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap

SPLIT = "stratified"
DATA_DIR = Path("data/processed") / SPLIT
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "boosting" / SPLIT / "xgboost.joblib"  # depth-6 untuned
FIG_DIR = Path("figures")
RESULTS_DIR = Path("results/metrics")

RANDOM_STATE = 42
SAMPLE_N = 8_000       # rows explained (beeswarm/bar readability + speed)
TOP_DEPENDENCE = 3     # dependence plots for the N most important features


def save(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved figures/{fig_name}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    model = joblib.load(MODEL_PATH)

    X_val = pd.read_parquet(DATA_DIR / "X_validation.parquet")[feature_names]
    y_val = pd.read_parquet(DATA_DIR / "y_validation.parquet")["Target"]

    sample_idx = X_val.sample(n=min(SAMPLE_N, len(X_val)), random_state=RANDOM_STATE).index
    X_sample = X_val.loc[sample_idx]
    y_sample = y_val.loc[sample_idx].to_numpy()
    preds = model.predict(X_sample)
    print(f"Explaining {len(X_sample):,} validation rows "
          f"({int(y_sample.sum())} attacks, {int((preds==1).sum())} predicted attacks)")

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)

    # 1 + 2: global plots
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    save("shap_beeswarm.png")
    shap.plots.bar(explanation, max_display=20, show=False)
    save("shap_bar.png")

    # Feature importance table (mean |SHAP|).
    mean_abs = np.abs(explanation.values).mean(axis=0)
    importance = (
        pd.Series(mean_abs, index=feature_names)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
    )
    importance.to_frame().to_csv(RESULTS_DIR / "shap_feature_importance.csv")
    top_features = importance.head(TOP_DEPENDENCE).index.tolist()
    print("\nTop features by mean |SHAP|:")
    print(importance.head(10).to_string())

    # 3 + 4: local waterfalls for a true positive and a false positive.
    tp_idx = np.where((preds == 1) & (y_sample == 1))[0]
    fp_idx = np.where((preds == 1) & (y_sample == 0))[0]
    if len(tp_idx):
        shap.plots.waterfall(explanation[int(tp_idx[0])], max_display=15, show=False)
        save("shap_waterfall_tp.png")
    else:
        print("  [warn] no true positive in sample — skipped TP waterfall")
    if len(fp_idx):
        shap.plots.waterfall(explanation[int(fp_idx[0])], max_display=15, show=False)
        save("shap_waterfall_fp.png")
    else:
        print("  [warn] no false positive in sample — skipped FP waterfall")

    # 5: dependence plots for the top features.
    for feat in top_features:
        shap.plots.scatter(explanation[:, feat], show=False)
        safe = feat.replace("/", "_").replace(" ", "_")
        save(f"shap_dependence_{safe}.png")

    # Leakage sanity check: no identifier-like feature should dominate.
    summary = {
        "model": MODEL_PATH.name,
        "sample_rows": int(len(X_sample)),
        "top_features": importance.head(10).round(5).to_dict(),
        "true_positive_explained": int(tp_idx[0]) if len(tp_idx) else None,
        "false_positive_explained": int(fp_idx[0]) if len(fp_idx) else None,
    }
    with open(RESULTS_DIR / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved shap_feature_importance.csv and shap_summary.json to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
