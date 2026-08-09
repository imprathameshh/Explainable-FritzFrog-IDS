"""Phase 7 — merge cleaned per-file Parquet into one processed dataset."""
from pathlib import Path
import json

import pandas as pd

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/metrics")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INTERIM_DIR.glob("*_cleaned.parquet"))
    if not files:
        raise FileNotFoundError(
            "No cleaned Parquet files in data/interim. "
            "Run 'python src/02_preprocess.py' first."
        )

    frames = []
    per_file_columns = {}
    for file_path in files:
        frame = pd.read_parquet(file_path)
        frame["SourceFile"] = file_path.name
        per_file_columns[file_path.name] = set(frame.columns)
        frames.append(frame)

    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    common_columns = sorted(common)

    all_columns = set().union(*per_file_columns.values())
    excluded_columns = sorted(all_columns - set(common_columns))

    aligned = [frame[common_columns] for frame in frames]
    combined = pd.concat(aligned, ignore_index=True)

    # Defensive: ensure feature columns are numeric before writing Parquet.
    # Guards against interim files with object dtype caused by embedded
    # repeated-header rows / stray tokens in the source CSVs.
    non_feature = {"Label", "SourceFile"}
    coerced_columns = []
    for col in combined.columns:
        if col in non_feature or combined[col].dtype != object:
            continue
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        coerced_columns.append(col)

    output = PROCESSED_DIR / "combined_cleaned.parquet"
    combined.to_parquet(output, index=False)

    report = {
        "files_combined": [f.name for f in files],
        "common_columns": common_columns,
        "excluded_columns": excluded_columns,
        "coerced_columns": coerced_columns,
        "combined_shape": list(combined.shape),
    }
    with open(RESULTS_DIR / "combine_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Combined shape:", combined.shape)
    if excluded_columns:
        print("Columns excluded during alignment:", excluded_columns)
    if coerced_columns:
        print(f"Columns coerced to numeric ({len(coerced_columns)}):", coerced_columns)
    print("Saved:", output)


if __name__ == "__main__":
    main()
