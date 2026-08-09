import numpy as np
import pandas as pd

from cleaning import clean_dataframe


def test_column_names_stripped(raw_frame):
    cleaned, _ = clean_dataframe(raw_frame)
    assert all(col == col.strip() for col in cleaned.columns)
    assert "Dst Port" in cleaned.columns


def test_duplicates_removed(raw_frame):
    _, report = clean_dataframe(raw_frame)
    assert report["duplicates_removed"] == 1


def test_inf_and_nan_rows_dropped(raw_frame):
    cleaned, report = clean_dataframe(raw_frame)
    assert report["rows_dropped_for_na_or_inf"] == 2
    numeric = cleaned.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    assert not cleaned.isna().any().any()


def test_identifier_columns_dropped_when_present(raw_frame):
    cleaned, report = clean_dataframe(raw_frame)
    assert "Timestamp" in report["identifier_columns_dropped"]
    assert "Timestamp" not in cleaned.columns


def test_identifier_columns_untouched_when_absent():
    df = pd.DataFrame({"Dst Port": [1, 2], "Label": ["Benign", "Bot"]})
    cleaned, report = clean_dataframe(df)
    assert report["identifier_columns_dropped"] == []
    assert list(cleaned.columns) == ["Dst Port", "Label"]


def test_report_arithmetic_consistent(raw_frame):
    _, report = clean_dataframe(raw_frame)
    assert (
        report["rows_before"]
        - report["duplicates_removed"]
        - report["rows_dropped_for_na_or_inf"]
        == report["rows_after"]
    )


def test_row_loss_pct(raw_frame):
    _, report = clean_dataframe(raw_frame)
    # 5 rows -> 2 rows == 60% lost.
    assert report["row_loss_pct"] == 60.0


def test_empty_after_cleaning_does_not_crash():
    df = pd.DataFrame(
        {"Dst Port": [np.nan, np.inf], "Label": ["Benign", "Bot"]}
    )
    cleaned, report = clean_dataframe(df)
    assert len(cleaned) == 0
    assert report["rows_after"] == 0
    assert report["row_loss_pct"] == 100.0


def test_object_numeric_columns_coerced():
    # Simulates the CSE-CIC-IDS2018 defect: a numeric column read as strings
    # (dtype object) because an embedded repeated-header row put the literal
    # header token in the data. Coercion must yield a numeric column and drop
    # the offending row, so per-file Parquet merges without dtype conflicts.
    df = pd.DataFrame(
        {
            "ACK Flag Cnt": ["0", "ACK Flag Cnt", "1"],  # header token embedded
            "Flow Duration": ["100", "200", "300"],
            "Label": ["Benign", "Benign", "Bot"],
        }
    )
    cleaned, report = clean_dataframe(df)
    assert report["cells_coerced_to_nan"] == 1
    assert len(cleaned) == 2  # the header-token row is dropped
    assert cleaned["ACK Flag Cnt"].dtype.kind in "iuf"
    assert cleaned["Flow Duration"].dtype.kind in "iuf"


def test_input_frame_not_mutated(raw_frame):
    before_cols = list(raw_frame.columns)
    before_len = len(raw_frame)
    clean_dataframe(raw_frame)
    assert list(raw_frame.columns) == before_cols
    assert len(raw_frame) == before_len
