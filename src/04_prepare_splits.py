"""Phase 11 — build train/validation/test splits (two evaluation tracks).

PRIMARY  — stratified 70/15/15 (stratified on the 15-class label so every
           attack family, including rare Web/SQLi, appears in all three sets).
           Gives representative headline metrics and clean model selection.
           Written to data/processed/stratified/.

SECONDARY — attack-preserving DAY-BASED holdout (whole capture days assigned to
           each split; guide §14). A cross-day / hard-family generalisation
           experiment. Written to data/processed/dayheld/.

Target is binary: benign = 0, any attack = 1. ``SourceFile`` is dropped from the
features (it encodes the capture day and would leak). No scaling here — that is
fit on training data only, downstream.

The stratified split is written by STREAMING the combined Parquet in batches so
peak memory stays bounded (the full 15.7M-row frame never lives in RAM at once).

Run from the project root: python src/04_prepare_splits.py
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/processed/combined_cleaned.parquet")
PROCESSED_DIR = Path("data/processed")
STRATIFIED_DIR = PROCESSED_DIR / "stratified"
DAYHELD_DIR = PROCESSED_DIR / "dayheld"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/metrics")

LABEL_COLUMN = "Label"
SOURCE_COLUMN = "SourceFile"
RANDOM_STATE = 42
BATCH_ROWS = 500_000  # streaming batch size for the stratified split

# Day assignment for the secondary (attack-preserving) day-based holdout.
VAL_DAYS = ["Thursday-01-03-2018"]
TEST_DAYS = ["Friday-23-02-2018"]

SPLITS = ["train", "validation", "test"]


def day_key(source_file: str) -> str:
    return source_file.split("_TrafficForML")[0]


def to_target(labels: pd.Series) -> pd.Series:
    return labels.astype(str).str.strip().str.lower().ne("benign").astype(int)


def feature_columns_from_schema() -> list[str]:
    schema = pq.ParquetFile(DATA_FILE).schema_arrow
    return [
        f.name
        for f in schema
        if f.name not in (LABEL_COLUMN, SOURCE_COLUMN)
        and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))
    ]


def summarise(y: pd.Series, extra: dict) -> dict:
    attack = int(y.sum())
    return {
        **extra,
        "rows": int(len(y)),
        "attack": attack,
        "benign": int(len(y) - attack),
        "attack_fraction": round(float(y.mean()), 6),
    }


# --------------------------------------------------------------------------- #
# PRIMARY: stratified split, written by streaming to bound memory.
# --------------------------------------------------------------------------- #
def stratified_assignment(labels: np.ndarray) -> np.ndarray:
    """Return an int8 array: 0=train, 1=validation, 2=test (70/15/15)."""
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=RANDOM_STATE, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=labels[temp_idx]
    )
    assign = np.empty(len(labels), dtype=np.int8)
    assign[train_idx] = 0
    assign[val_idx] = 1
    assign[test_idx] = 2
    return assign


def write_stratified(feature_names: list[str]) -> dict:
    STRATIFIED_DIR.mkdir(parents=True, exist_ok=True)

    labels = (
        pd.read_parquet(DATA_FILE, columns=[LABEL_COLUMN])[LABEL_COLUMN]
        .astype(str)
        .str.strip()
        .to_numpy()
    )
    assign = stratified_assignment(labels)
    del labels

    x_writers: dict[int, pq.ParquetWriter] = {}
    y_writers: dict[int, pq.ParquetWriter] = {}
    counts = {k: {"rows": 0, "attack": 0} for k in range(3)}

    reader = pq.ParquetFile(DATA_FILE)
    offset = 0
    for batch in reader.iter_batches(batch_size=BATCH_ROWS):
        frame = batch.to_pandas()
        n = len(frame)
        part = assign[offset : offset + n]
        offset += n
        target = to_target(frame[LABEL_COLUMN])

        for k in range(3):
            mask = part == k
            if not mask.any():
                continue
            x_tbl = pa.Table.from_pandas(
                frame.loc[mask, feature_names], preserve_index=False
            )
            y_tbl = pa.Table.from_pandas(
                target[mask].to_frame("Target"), preserve_index=False
            )
            split = SPLITS[k]
            if k not in x_writers:
                x_writers[k] = pq.ParquetWriter(
                    STRATIFIED_DIR / f"X_{split}.parquet", x_tbl.schema
                )
                y_writers[k] = pq.ParquetWriter(
                    STRATIFIED_DIR / f"y_{split}.parquet", y_tbl.schema
                )
            x_writers[k].write_table(x_tbl)
            y_writers[k].write_table(y_tbl)
            counts[k]["rows"] += int(mask.sum())
            counts[k]["attack"] += int(target[mask].sum())

    for w in list(x_writers.values()) + list(y_writers.values()):
        w.close()

    report = {}
    for k, split in enumerate(SPLITS):
        rows, attack = counts[k]["rows"], counts[k]["attack"]
        report[split] = {
            "rows": rows,
            "attack": attack,
            "benign": rows - attack,
            "attack_fraction": round(attack / rows, 6) if rows else 0.0,
        }
        print(f"  stratified {split}: rows={rows:,} attack={attack:,} "
              f"({100 * attack / rows:.3f}%)")
    return report


# --------------------------------------------------------------------------- #
# SECONDARY: attack-preserving day-based holdout (per-split parquet filters).
# --------------------------------------------------------------------------- #
def write_dayheld(feature_names: list[str]) -> dict:
    DAYHELD_DIR.mkdir(parents=True, exist_ok=True)

    source_files = (
        pd.read_parquet(DATA_FILE, columns=[SOURCE_COLUMN])[SOURCE_COLUMN]
        .unique()
        .tolist()
    )
    by_day = {day_key(sf): sf for sf in source_files}
    missing = [d for d in VAL_DAYS + TEST_DAYS if d not in by_day]
    if missing:
        raise ValueError(f"Configured val/test days not found: {missing}")

    val_files = [by_day[d] for d in VAL_DAYS]
    test_files = [by_day[d] for d in TEST_DAYS]
    train_files = [sf for sf in source_files if sf not in val_files + test_files]
    split_files = {"train": train_files, "validation": val_files, "test": test_files}

    report = {"val_days": VAL_DAYS, "test_days": TEST_DAYS, "splits": {}}
    for split, files in split_files.items():
        frame = pd.read_parquet(DATA_FILE, filters=[(SOURCE_COLUMN, "in", files)])
        y = to_target(frame[LABEL_COLUMN])
        frame[feature_names].to_parquet(DAYHELD_DIR / f"X_{split}.parquet", index=False)
        y.to_frame("Target").to_parquet(DAYHELD_DIR / f"y_{split}.parquet", index=False)
        report["splits"][split] = summarise(y, {"days": [day_key(f) for f in files]})
        print(f"  dayheld {split}: {[day_key(f) for f in files]} "
              f"rows={len(y):,} attack={int(y.sum()):,} ({100 * y.mean():.3f}%)")
        del frame, y
    return report


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run 01->02->03 first.")

    feature_names = feature_columns_from_schema()
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
    print(f"Features: {len(feature_names)}")

    print("\n[PRIMARY] stratified 70/15/15:")
    stratified_report = write_stratified(feature_names)

    print("\n[SECONDARY] attack-preserving day-based holdout:")
    dayheld_report = write_dayheld(feature_names)

    report = {
        "n_features": len(feature_names),
        "stratified": stratified_report,
        "dayheld": dayheld_report,
    }
    with open(RESULTS_DIR / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nStratified split -> {STRATIFIED_DIR}")
    print(f"Day-held split   -> {DAYHELD_DIR}")
    print("Saved split_report.json and feature_names.joblib")


if __name__ == "__main__":
    main()
