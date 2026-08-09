"""Reusable data-cleaning logic for CSE-CIC-IDS2018 CSV flow data.

Kept import-safe and side-effect-free so it can be unit tested without any
file I/O or dataset download. ``02_preprocess.py`` imports ``clean_dataframe``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Identifier / leakage-prone columns. Dropped only when present. Retaining
# these risks the model memorising the capture environment rather than
# learning general flow behaviour (guide phases 6 and 10).
IDENTIFIER_COLUMNS = ["Timestamp", "Flow ID", "Src IP", "Dst IP"]


def clean_dataframe(
    df: pd.DataFrame,
    identifier_columns: list[str] | None = None,
    label_column: str = "Label",
) -> tuple[pd.DataFrame, dict]:
    """Clean a single raw flow DataFrame.

    Steps, in order:
      1. Strip whitespace from column names.
      2. Drop exact duplicate rows.
      3. Drop identifier columns when present (before NaN handling, so their
         non-numeric values don't force whole rows to be dropped).
      4. Coerce every non-label column to numeric. CSE-CIC-IDS2018 CSVs contain
         embedded repeated-header rows and stray tokens that make numeric
         columns parse as ``object``/strings; ``to_numeric(errors="coerce")``
         turns those into NaN (removed in the next step) and gives every file a
         consistent numeric dtype so per-file Parquet files merge cleanly.
      5. Replace +/-inf with NaN, then drop rows containing any NaN.

    Returns the cleaned frame and a report describing what changed. The input
    frame is not mutated. The report guarantees the invariant::

        rows_before - duplicates_removed - rows_dropped_for_na_or_inf == rows_after
    """
    if identifier_columns is None:
        identifier_columns = IDENTIFIER_COLUMNS

    df = df.copy()

    rows_before = len(df)
    columns_before = len(df.columns)

    # 1. Column names first, so dedup and label logic see canonical names.
    df.columns = df.columns.str.strip()

    # 2. Exact duplicates (counted after the column strip).
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates()
    rows_after_dedup = len(df)

    # 3. Identifier columns, dropped only when present — before numeric
    #    coercion and NaN handling.
    columns_dropped = [c for c in identifier_columns if c in df.columns]
    if columns_dropped:
        df = df.drop(columns=columns_dropped)

    # 4. Coerce non-label columns to numeric; count values lost to coercion.
    cells_coerced_to_nan = 0
    for col in df.columns:
        if col == label_column or is_numeric_dtype(df[col]):
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        cells_coerced_to_nan += int(coerced.isna().sum() - df[col].isna().sum())
        df[col] = coerced

    # 5. Infinities -> NaN -> drop. Record per-column NaN before dropping.
    df = df.replace([np.inf, -np.inf], np.nan)
    na_per_column = {
        col: int(count)
        for col, count in df.isna().sum().items()
        if count > 0
    }
    rows_dropped_for_na_or_inf = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    rows_after_na = len(df)

    rows_after = len(df)
    row_loss_pct = (
        round(100.0 * (rows_before - rows_after) / rows_before, 4)
        if rows_before
        else 0.0
    )

    report = {
        "rows_before": rows_before,
        "columns_before": columns_before,
        "duplicates_removed": duplicate_count,
        "rows_after_dedup": rows_after_dedup,
        "identifier_columns_dropped": columns_dropped,
        "cells_coerced_to_nan": cells_coerced_to_nan,
        "na_per_column": na_per_column,
        "rows_dropped_for_na_or_inf": rows_dropped_for_na_or_inf,
        "rows_after_na_drop": rows_after_na,
        "rows_after": rows_after,
        "columns_after": len(df.columns),
        "row_loss_pct": row_loss_pct,
    }

    return df, report
