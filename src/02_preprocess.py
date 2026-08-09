"""Phase 6 — clean each raw CSV and write per-file cleaned Parquet.

Precondition: ``data/raw/`` populated (run ``bash src/00_download_data.sh``).
Reads raw CSVs, applies ``cleaning.clean_dataframe``, writes cleaned frames to
``data/interim/``, and saves an aggregated report to
``results/metrics/cleaning_report.json``. Raw data is never modified.

Run from the project root: python src/02_preprocess.py
"""
from pathlib import Path
import json

import pandas as pd

from cleaning import clean_dataframe

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
RESULTS_DIR = Path("results/metrics")


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files in data/raw. Download the dataset first: "
            "run 'bash src/00_download_data.sh' (see README Prerequisites)."
        )

    all_reports = []
    for file_path in csv_files:
        print(f"Processing {file_path.name}")
        df = pd.read_csv(file_path, low_memory=False)
        cleaned, report = clean_dataframe(df)

        output_path = INTERIM_DIR / f"{file_path.stem}_cleaned.parquet"
        cleaned.to_parquet(output_path, index=False)

        report["source_file"] = file_path.name
        report["output_file"] = output_path.name
        all_reports.append(report)
        print(
            f"  saved {output_path.name} "
            f"({report['rows_after']} rows, {report['row_loss_pct']}% lost)"
        )

    report_path = RESULTS_DIR / "cleaning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"Cleaning report written to {report_path}")


if __name__ == "__main__":
    main()
