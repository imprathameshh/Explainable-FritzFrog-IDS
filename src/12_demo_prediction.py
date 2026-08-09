"""Optional deployment demo (guide §30) — score a CSV of network flows.

A lightweight command-line demonstration: load the saved final XGBoost model,
take a CICFlowMeter-style CSV, align it to the training features, and print a
benign/attack prediction with probability per flow — plus, optionally, the top
SHAP contributors for each flagged flow.

No live malware is involved; this scores pre-extracted flow features. Model
correctness and research evaluation remain the priority over this prototype.

Usage:
  python src/12_demo_prediction.py path/to/flows.csv
  python src/12_demo_prediction.py path/to/flows.csv --track dayheld --explain --output preds.csv
"""
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")


def prepare(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Align an incoming CSV to the model's features (inference-time cleaning)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required feature(s): {missing[:5]}...")
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Score network flows for FritzFrog-like behaviour.")
    parser.add_argument("csv", type=Path, help="CSV of CICFlowMeter flow features")
    parser.add_argument("--track", default="stratified", choices=["stratified", "dayheld"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--explain", action="store_true", help="show top SHAP contributors")
    parser.add_argument("--output", type=Path, help="write predictions to this CSV")
    args = parser.parse_args()

    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    model_path = MODELS_DIR / "final" / args.track / "xgboost_final.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found — run 10_train_final_model.py first.")
    model = joblib.load(model_path)

    df = pd.read_csv(args.csv, low_memory=False)
    X = prepare(df, feature_names)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    result = pd.DataFrame({
        "prediction": np.where(pred == 1, "ATTACK", "benign"),
        "attack_probability": proba.round(4),
    })

    if args.explain:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X).values
        for i in np.where(pred == 1)[0]:
            top = pd.Series(shap_values[i], index=feature_names).sort_values(
                key=np.abs, ascending=False
            ).head(3)
            drivers = ", ".join(f"{k}={X.iloc[i][k]:.4g} (SHAP {v:+.3f})" for k, v in top.items())
            result.loc[i, "top_drivers"] = drivers

    n_attack = int(pred.sum())
    print(f"Scored {len(X):,} flows | {n_attack:,} flagged as ATTACK "
          f"({100 * n_attack / len(X):.2f}%) | track={args.track}, threshold={args.threshold}")
    print(result.head(20).to_string())
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
