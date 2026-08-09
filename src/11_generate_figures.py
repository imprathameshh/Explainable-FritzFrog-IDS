"""Phase 23 — generate dissertation figures from saved artifacts.

Reads metrics/predictions produced by 05/06/09/10 and writes 300-DPI figures
to figures/. Each figure is skipped (with a message) if its source is missing,
so the script is safe to run at any pipeline stage.

Figures:
  - confusion matrix (final test, per track)
  - ROC curve and PR curve (final test, per track)
  - model comparison bar chart (baselines + boosting)
  - feature-selection: PR-AUC vs feature count
  - training / inference time comparison

Run from the project root: python src/11_generate_figures.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

RESULTS_DIR = Path("results/metrics")
PRED_DIR = Path("results/predictions")
FIG_DIR = Path("figures")
TRACKS = ["stratified", "dayheld"]


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved figures/{name}")


def confusion_figures():
    path = RESULTS_DIR / "final_test_metrics.json"
    if not path.exists():
        print("  [skip] final_test_metrics.json missing (run 10)")
        return
    metrics = json.loads(path.read_text())
    for track, m in metrics.items():
        cm = m["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Benign", "Attack"])
        ax.set_yticks([0, 1], ["Benign", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {track} test")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        save(f"confusion_matrix_{track}.png")


def curve_figures():
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(5.5, 5))
    any_curve = False
    for track in TRACKS:
        p = PRED_DIR / f"final_test_predictions_{track}.parquet"
        if not p.exists():
            continue
        any_curve = True
        df = pd.read_parquet(p)
        y, proba = df["y_true"].to_numpy(), df["proba"].to_numpy()
        fpr, tpr, _ = roc_curve(y, proba)
        ax_roc.plot(fpr, tpr, label=f"{track} (AUC={auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y, proba)
        ax_pr.plot(rec, prec, label=f"{track} (AP={average_precision_score(y, proba):.3f})")
    if not any_curve:
        plt.close(fig_roc); plt.close(fig_pr)
        print("  [skip] no predictions (run 10)")
        return
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC — final test"); ax_roc.legend()
    plt.figure(fig_roc.number); save("roc_curve.png")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall — final test"); ax_pr.legend()
    plt.figure(fig_pr.number); save("precision_recall_curve.png")


def model_comparison_figure():
    frames = []
    for f in ["baseline_comparison_stratified.csv", "model_comparison_stratified.csv"]:
        p = RESULTS_DIR / f
        if p.exists():
            frames.append(pd.read_csv(p, index_col=0))
    if not frames:
        print("  [skip] no comparison CSVs (run 05/06)")
        return
    tab = pd.concat(frames).sort_values("pr_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tab)); w = 0.4
    ax.bar(x - w / 2, tab["f1"], w, label="F1")
    ax.bar(x + w / 2, tab["pr_auc"], w, label="PR-AUC")
    ax.set_xticks(x, tab.index, rotation=30, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.set_title("Model comparison (validation)"); ax.legend()
    save("model_comparison.png")
    # timing
    if "train_time_s" in tab:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, tab["train_time_s"])
        ax.set_xticks(x, tab.index, rotation=30, ha="right")
        ax.set_ylabel("Training time (s)"); ax.set_title("Training time by model")
        save("training_time.png")


def feature_selection_figure():
    p = RESULTS_DIR / "feature_selection_comparison_stratified.csv"
    if not p.exists():
        print("  [skip] feature_selection_comparison missing (run 09)")
        return
    tab = pd.read_csv(p, index_col=0).sort_values("n_features")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(tab["n_features"], tab["pr_auc"], "o-", label="PR-AUC")
    ax.plot(tab["n_features"], tab["f1"], "s--", label="F1")
    for name, r in tab.iterrows():
        ax.annotate(name, (r["n_features"], r["pr_auc"]), fontsize=6,
                    textcoords="offset points", xytext=(3, 4))
    ax.set_xlabel("Number of features"); ax.set_ylabel("Score")
    ax.set_title("Performance vs feature count"); ax.legend()
    save("rfe_comparison.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    confusion_figures()
    curve_figures()
    model_comparison_figure()
    feature_selection_figure()
    print("Done.")


if __name__ == "__main__":
    main()
