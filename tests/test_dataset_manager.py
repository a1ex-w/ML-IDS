### TC-01: DatasetManager (pipeline/dataset_manager.py)
# Checks the cleaning steps on the fake dataset, where the number of bad rows is known.

### Imports
import numpy as np
import pandas as pd
import pytest

from conftest import EXPECTED_CLEAN_ROWS, EXPECTED_FEATURES, FAKE_FILE_COUNT, ATTACKS_PER_FILE, BENIGN_PER_FILE


@pytest.fixture(scope="module")
def cleaned_df(workspace):
    return pd.read_csv(workspace["dir"] / "data" / "cleaned.csv")


def test_tc01_all_files_merged(workspace):
    output = workspace["step_output"]["dataset_manager"]
    assert f"Found {FAKE_FILE_COUNT} CSV files" in output


def test_tc01_headers_stripped(cleaned_df):
    assert all(col == col.strip() for col in cleaned_df.columns)


def test_tc01_duplicate_fwd_header_length_dropped(cleaned_df):
    assert "Fwd Header Length.1" not in cleaned_df.columns
    assert "Fwd Header Length" in cleaned_df.columns


def test_tc01_constant_columns_dropped(cleaned_df):
    assert "Bwd PSH Flags" not in cleaned_df.columns


def test_tc01_expected_feature_columns(cleaned_df):
    feature_columns = [col for col in cleaned_df.columns if col not in ["Attack Type", "is_attack"]]
    assert feature_columns == EXPECTED_FEATURES


def test_tc01_no_inf_or_nan(cleaned_df):
    numeric_values = cleaned_df.drop(columns=["Attack Type"]).to_numpy(dtype=float)
    assert not np.isnan(numeric_values).any()
    assert not np.isinf(numeric_values).any()


def test_tc01_negative_rows_dropped_except_init_win(cleaned_df):
    check_columns = [col for col in EXPECTED_FEATURES if col not in ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]]
    assert (cleaned_df[check_columns] >= 0).all().all()
    assert (cleaned_df["Init_Win_bytes_forward"] == -1).any()  # -1 is valid and should be kept


def test_tc01_duplicates_dropped(cleaned_df):
    assert cleaned_df.duplicated().sum() == 0


def test_tc01_exact_row_count(cleaned_df):
    # every planted bad row (inf, NaN, negative, duplicate) removed, every good row kept
    assert len(cleaned_df) == EXPECTED_CLEAN_ROWS


def test_tc01_labels(cleaned_df):
    expected_types = {"BENIGN", "DDoS", "PortScan", "Web Attack - Brute Force"}  # garbled dash fixed
    assert set(cleaned_df["Attack Type"]) == expected_types
    assert set(cleaned_df["is_attack"]) == {0, 1}
    assert (cleaned_df.loc[cleaned_df["Attack Type"] == "BENIGN", "is_attack"] == 0).all()
    assert (cleaned_df.loc[cleaned_df["Attack Type"] != "BENIGN", "is_attack"] == 1).all()


def test_tc01_class_counts(cleaned_df):
    assert (cleaned_df["Attack Type"] == "BENIGN").sum() == BENIGN_PER_FILE * FAKE_FILE_COUNT
    assert (cleaned_df["Attack Type"] == "DDoS").sum() == ATTACKS_PER_FILE["DDoS"] * FAKE_FILE_COUNT
