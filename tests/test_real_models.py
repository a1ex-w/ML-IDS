### Requirements and performance tests on the REAL trained models
# Uses the models/ folder made by running the pipeline on the full CICIDS2017 data.
# Skipped automatically if models/ doesn't exist (e.g. a fresh clone).
# Run with -s to see the printed performance table for the Milestone 4 document.

### Imports
import json
import time
import tracemalloc
from pathlib import Path

import joblib
import pytest
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODEL_KEYS = ["random_forest", "logistic_regression", "linear_svm"]

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "metrics.json").exists(), reason="real models/ not found; run the pipeline first")


@pytest.fixture(scope="module")
def real_metrics():
    with open(MODELS_DIR / "metrics.json") as metrics_file:
        return json.load(metrics_file)


### NFR-1: accuracy over 80% for every model
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_nfr1_accuracy_over_80_percent(real_metrics, model_key):
    assert real_metrics[model_key]["accuracy"] > 0.80


### Performance: inference time, memory, and file size on the real test set
def test_performance_report(real_metrics):
    test_data = joblib.load(MODELS_DIR / "test_data.pkl")
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    print(f"\n\nReal test set: {len(y_test):,} rows, {X_test.shape[1]} features")
    print(f"{'Model':<22}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'AUC':>8}{'FN':>7}{'Infer (s)':>11}{'Rows/s':>11}{'Peak MB':>9}{'File MB':>9}")

    for model_key in MODEL_KEYS:
        model_path = MODELS_DIR / f"{model_key}.pkl"
        model = joblib.load(model_path)

        # time and measure memory for scoring the whole test set
        tracemalloc.start()
        start_time = time.perf_counter()
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = expit(model.decision_function(X_test))
        infer_seconds = time.perf_counter() - start_time
        peak_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        tracemalloc.stop()

        # recomputed accuracy should match what the dashboard shows
        accuracy = ((y_score >= 0.5).astype(int) == y_test).mean()
        assert round(accuracy, 4) == real_metrics[model_key]["accuracy"]

        m = real_metrics[model_key]
        file_mb = model_path.stat().st_size / 1024 / 1024
        print(f"{m['name']:<22}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}{m['roc_auc']:>8.4f}{m['fn']:>7,}{infer_seconds:>11.3f}{len(y_test) / infer_seconds:>11,.0f}{peak_mb:>9.1f}{file_mb:>9.1f}")
