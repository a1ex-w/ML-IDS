### TC-04: Evaluator (pipeline/evaluator.py)
# Recomputes the metrics with scikit-learn and checks they match what the Evaluator saved.

### Imports
import json

import joblib
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_KEYS = ["random_forest", "logistic_regression", "linear_svm"]


@pytest.fixture(scope="module")
def saved(workspace):
    models_dir = workspace["dir"] / "models"
    with open(models_dir / "metrics.json") as metrics_file:
        metrics = json.load(metrics_file)
    with open(models_dir / "results_explorer.json") as results_file:
        explorer = json.load(results_file)
    test_data = joblib.load(models_dir / "test_data.pkl")
    scores = joblib.load(models_dir / "test_scores.pkl")
    return {"dir": workspace["dir"], "metrics": metrics, "explorer": explorer, "test_data": test_data, "scores": scores}


def test_tc04_metrics_saved_for_all_models(saved):
    assert set(saved["metrics"]) == set(MODEL_KEYS)
    for model_key in MODEL_KEYS:
        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0 <= saved["metrics"][model_key][metric_name] <= 1


def test_tc04_metrics_match_sklearn(saved):
    y_test = saved["test_data"]["y_test"]
    for model_key in MODEL_KEYS:
        model = joblib.load(saved["dir"] / "models" / f"{model_key}.pkl")
        y_pred = model.predict(saved["test_data"]["X_test"])
        model_metrics = saved["metrics"][model_key]
        assert model_metrics["accuracy"] == pytest.approx(accuracy_score(y_test, y_pred), abs=1e-4)
        assert model_metrics["precision"] == pytest.approx(precision_score(y_test, y_pred), abs=1e-4)
        assert model_metrics["recall"] == pytest.approx(recall_score(y_test, y_pred), abs=1e-4)
        assert model_metrics["f1"] == pytest.approx(f1_score(y_test, y_pred), abs=1e-4)


def test_tc04_confusion_matrix_adds_up(saved):
    test_size = len(saved["test_data"]["y_test"])
    for model_key in MODEL_KEYS:
        model_metrics = saved["metrics"][model_key]
        assert model_metrics["tn"] + model_metrics["fp"] + model_metrics["fn"] + model_metrics["tp"] == test_size


def test_tc04_svm_score_matches_predict_at_half(saved):
    # sigmoid on decision_function should give the same labels as predict() at 0.5
    model = joblib.load(saved["dir"] / "models" / "linear_svm.pkl")
    y_pred = model.predict(saved["test_data"]["X_test"])
    score_pred = (saved["scores"]["linear_svm"] >= 0.5).astype(int)
    assert (y_pred == score_pred).all()


def test_tc04_recall_drops_as_threshold_rises(saved):
    for model_key in MODEL_KEYS:
        recalls = [row["recall"] for row in saved["explorer"]["threshold_metrics"][model_key]]
        assert all(later <= earlier for earlier, later in zip(recalls, recalls[1:]))


def test_tc04_threshold_list(saved):
    thresholds = saved["explorer"]["thresholds"]
    assert thresholds[0] == 0.05 and thresholds[-1] == 0.95 and len(thresholds) == 19


def test_tc04_plots_created(saved):
    plots_dir = saved["dir"] / "static" / "plots"
    for model_key in MODEL_KEYS:
        assert (plots_dir / f"cm_{model_key}.png").exists()
    assert (plots_dir / "feature_importance.png").exists()


def test_tc04_models_learn_on_fake_data(saved):
    # sanity check that training actually worked (real-data numbers are in test_real_models.py)
    assert saved["metrics"]["random_forest"]["accuracy"] > 0.9
