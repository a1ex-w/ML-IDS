### TC-02 / TC-03: ModelTrainer (pipeline/model_trainer.py)
# Checks the split, scaling, training, and saved files.

### Imports
import joblib
import numpy as np
import pytest

from conftest import EXPECTED_CLEAN_ROWS, EXPECTED_FEATURES

MODEL_FILES = ["random_forest.pkl", "logistic_regression.pkl", "linear_svm.pkl"]


@pytest.fixture(scope="module")
def models_dir(workspace):
    return workspace["dir"] / "models"


def test_tc03_all_files_saved(models_dir):
    for file_name in MODEL_FILES + ["scaler.pkl", "feature_columns.pkl", "test_data.pkl"]:
        assert (models_dir / file_name).exists(), f"missing {file_name}"


def test_tc03_split_sizes(models_dir):
    test_data = joblib.load(models_dir / "test_data.pkl")
    sample_rows = int(EXPECTED_CLEAN_ROWS * 0.2)  # 20% stratified sample
    test_rows = len(test_data["y_test"])
    assert abs(test_rows - sample_rows * 0.2) <= 2  # 80/20 split


def test_tc03_test_split_has_both_classes(models_dir):
    test_data = joblib.load(models_dir / "test_data.pkl")
    assert set(np.unique(test_data["y_test"])) == {0, 1}


def test_tc02_scaler_fit_on_training_rows_only(models_dir):
    # no leakage: scaler should have seen exactly the training rows, not the test rows
    scaler = joblib.load(models_dir / "scaler.pkl")
    test_data = joblib.load(models_dir / "test_data.pkl")
    sample_rows = int(EXPECTED_CLEAN_ROWS * 0.2)
    train_rows = sample_rows - len(test_data["y_test"])
    assert scaler.n_samples_seen_ == train_rows


def test_tc02_scaler_matches_feature_list(models_dir):
    scaler = joblib.load(models_dir / "scaler.pkl")
    feature_columns = joblib.load(models_dir / "feature_columns.pkl")
    assert feature_columns == EXPECTED_FEATURES
    assert scaler.n_features_in_ == len(feature_columns)


def test_tc03_models_use_balanced_class_weight(models_dir):
    for file_name in MODEL_FILES:
        model = joblib.load(models_dir / file_name)
        assert model.class_weight == "balanced", file_name


def test_tc03_reloaded_models_predict(models_dir):
    test_data = joblib.load(models_dir / "test_data.pkl")
    for file_name in MODEL_FILES:
        model = joblib.load(models_dir / file_name)
        predictions = model.predict(test_data["X_test"])
        assert len(predictions) == len(test_data["y_test"])
        assert set(np.unique(predictions)) <= {0, 1}
