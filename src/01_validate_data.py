"""Phase 4 — validate the raw dataset before cleaning.

Precondition gate: confirms ``data/raw/`` is populated and each CSV is
readable, has a header, and carries a ``Label`` column. Exits non-zero on any
failure so it can gate the cleaning step in a pipeline.

Run from the project root: python src/01_validate_data.py
"""
from pathlib import Path
import sys

import pandas as pd

RAW_DIR = Path("data/raw")


def main() -> int:
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(
            "No CSV files found in data/raw. Download the dataset first: "
            "run 'bash src/00_download_data.sh' (see README Prerequisites).",
            file=sys.stderr,
        )
        return 1

    print(f"CSV files found: {len(csv_files)}")
    problems = 0
    for file_path in csv_files:
        try:
            sample = pd.read_csv(file_path, nrows=5, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - report and keep scanning
            print(f"  FAIL {file_path.name}: unreadable ({exc})")
            problems += 1
            continue

        columns = [c.strip() for c in sample.columns]
        has_label = "Label" in columns
        status = "ok" if has_label else "MISSING Label column"
        print(f"  {file_path.name}: {len(columns)} columns, {status}")
        if not has_label:
            problems += 1

    if problems:
        print(f"Validation finished with {problems} problem(s).", file=sys.stderr)
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
