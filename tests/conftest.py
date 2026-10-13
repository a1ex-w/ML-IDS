### Shared test setup
# Builds a small fake CICIDS2017-style dataset in a temp folder, then runs the real
# pipeline scripts on it (dataset_manager -> model_trainer -> evaluator -> make_sample_csv).
# This way the tests run in seconds and don't need the 845 MB dataset.

### Imports
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

### Fake dataset settings (known counts so tests can check exact numbers)
FAKE_FILE_COUNT = 3              # stands in for the 8 daily CSVs
BENIGN_PER_FILE = 600
ATTACKS_PER_FILE = {"DDoS": 100, "PortScan": 100, "Web Attack \x96 Brute Force": 60}  # \x96 = the garbled dash in the real data
INF_ROWS_PER_FILE = 3            # rows with inf in Flow Bytes/s (should be dropped)
NAN_ROWS_PER_FILE = 2            # rows with a blank value (should be dropped)
NEGATIVE_ROWS_PER_FILE = 2       # rows with a negative Flow Duration (should be dropped)
DUPLICATE_ROWS_PER_FILE = 5      # exact copies of good rows (should be dropped)

GOOD_ROWS_PER_FILE = BENIGN_PER_FILE + sum(ATTACKS_PER_FILE.values())
EXPECTED_CLEAN_ROWS = GOOD_ROWS_PER_FILE * FAKE_FILE_COUNT

# raw headers have leading spaces and a repeated Fwd Header Length, like the real CSVs
RAW_COLUMNS = [
    " Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
    "Flow Bytes/s", " Flow Packets/s", " Fwd Packet Length Max", " Bwd Packet Length Mean",
    " Fwd Header Length", " Bwd Header Length", " Fwd Header Length",
    "Init_Win_bytes_forward", " Init_Win_bytes_backward", " Bwd PSH Flags", " Label",
]
# after cleaning: headers stripped, duplicate Fwd Header Length dropped, constant Bwd PSH Flags dropped
EXPECTED_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packet Length Max", "Bwd Packet Length Mean",
    "Fwd Header Length", "Bwd Header Length",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("ml_ids_workspace")
    raw_dir = work_dir / "data" / "MachineLearningCVE"
    raw_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)

    ### Write fake raw CSVs
    for file_number in range(FAKE_FILE_COUNT):
        labels = ["BENIGN"] * BENIGN_PER_FILE
        for attack_name, attack_count in ATTACKS_PER_FILE.items():
            labels += [attack_name] * attack_count
        labels = np.array(labels)
        is_attack = (labels != "BENIGN").astype(float)
        row_count = len(labels)

        # attacks get shifted values so the models have something to learn
        feature_values = rng.uniform(1, 1000, size=(row_count, 11)) + is_attack[:, None] * rng.uniform(300, 900, size=(row_count, 11))
        init_win_fwd = rng.integers(0, 65535, size=row_count).astype(float)
        init_win_fwd[:20] = -1  # -1 means "no window observed" and must be kept

        good_df = pd.DataFrame({
            0: rng.integers(1, 65535, size=row_count),   # Destination Port
            1: feature_values[:, 0],                     # Flow Duration
            2: feature_values[:, 1],
            3: feature_values[:, 2],
            4: feature_values[:, 3],                     # Flow Bytes/s
            5: feature_values[:, 4],
            6: feature_values[:, 5],
            7: feature_values[:, 6],
            8: feature_values[:, 7],                     # Fwd Header Length
            9: feature_values[:, 8],
            10: feature_values[:, 7],                    # duplicate Fwd Header Length
            11: init_win_fwd,
            12: feature_values[:, 9],
            13: 0,                                       # Bwd PSH Flags, constant column
            14: labels,
        })

        # bad rows built from copies of good rows
        inf_rows = good_df.iloc[:INF_ROWS_PER_FILE].copy()
        inf_rows[4] = np.inf
        nan_rows = good_df.iloc[INF_ROWS_PER_FILE:INF_ROWS_PER_FILE + NAN_ROWS_PER_FILE].copy()
        nan_rows[4] = np.nan
        negative_rows = good_df.iloc[10:10 + NEGATIVE_ROWS_PER_FILE].copy()
        negative_rows[1] = -5.0
        duplicate_rows = good_df.iloc[30:30 + DUPLICATE_ROWS_PER_FILE].copy()

        raw_df = pd.concat([good_df, inf_rows, nan_rows, negative_rows, duplicate_rows], ignore_index=True)
        raw_df = raw_df.sample(frac=1, random_state=file_number)  # shuffle
        raw_df.columns = RAW_COLUMNS
        raw_df.to_csv(raw_dir / f"fake_day_{file_number}.csv", index=False, encoding="latin-1")

    ### Run the real pipeline scripts inside the temp folder
    step_times = {}
    step_output = {}
    for script_name in ["dataset_manager", "model_trainer", "evaluator", "make_sample_csv"]:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "pipeline" / f"{script_name}.py")],
            cwd=work_dir, capture_output=True, text=True,
        )
        step_times[script_name] = time.time() - start_time
        step_output[script_name] = result.stdout
        assert result.returncode == 0, f"{script_name}.py failed:\n{result.stderr}"

    return {"dir": work_dir, "step_times": step_times, "step_output": step_output}


@pytest.fixture(scope="session")
def flask_app(workspace):
    ### Load app.py from inside the temp folder so it picks up the fake models/
    original_dir = os.getcwd()
    os.chdir(workspace["dir"])
    start_time = time.time()
    app_spec = importlib.util.spec_from_file_location("ml_ids_app", REPO_ROOT / "app.py")
    app_module = importlib.util.module_from_spec(app_spec)
    sys.modules["ml_ids_app"] = app_module  # lets Flask find templates/ and static/ next to app.py
    app_spec.loader.exec_module(app_module)  # runs the startup code that loads models
    app_module.startup_seconds = time.time() - start_time
    os.chdir(original_dir)
    app_module.app.config["TESTING"] = True
    return app_module
