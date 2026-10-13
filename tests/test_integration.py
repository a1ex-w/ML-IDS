### IT-01 to IT-05: Integration tests
# Checks the handoffs between components, where mismatched columns or files would show up.

### Imports
import io
import json
import re

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit


### IT-01: DatasetManager -> ModelTrainer
def test_it01_cleaned_data_feeds_training(workspace):
    cleaned_df = pd.read_csv(workspace["dir"] / "data" / "cleaned.csv")
    feature_columns = joblib.load(workspace["dir"] / "models" / "feature_columns.pkl")
    cleaned_features = [col for col in cleaned_df.columns if col not in ["Attack Type", "is_attack"]]
    assert feature_columns == cleaned_features  # trainer used exactly the cleaned features, same order
    assert "Saved models" in workspace["step_output"]["model_trainer"]


### IT-02: ModelTrainer -> Evaluator
def test_it02_saved_models_reproduce_saved_metrics(workspace):
    models_dir = workspace["dir"] / "models"
    test_data = joblib.load(models_dir / "test_data.pkl")
    with open(models_dir / "metrics.json") as metrics_file:
        metrics = json.load(metrics_file)
    for model_key in ["random_forest", "logistic_regression", "linear_svm"]:
        model = joblib.load(models_dir / f"{model_key}.pkl")
        accuracy = (model.predict(test_data["X_test"]) == test_data["y_test"]).mean()
        assert round(accuracy, 4) == metrics[model_key]["accuracy"]


### IT-03: saved models and scaler -> DashboardController
def test_it03_app_loads_same_files(flask_app, workspace):
    models_dir = workspace["dir"] / "models"
    assert flask_app.feature_columns == joblib.load(models_dir / "feature_columns.pkl")
    assert set(flask_app.loaded_models) == {"random_forest", "logistic_regression", "linear_svm"}
    assert flask_app.scaler.n_features_in_ == len(flask_app.feature_columns)


def test_it03_app_predictions_match_offline_run(flask_app, workspace):
    # scale and predict the sample CSV by hand, then compare to what the upload page reports
    sample_df = pd.read_csv(workspace["dir"] / "static" / "sample_upload.csv")
    scaler = joblib.load(workspace["dir"] / "models" / "scaler.pkl")
    model = joblib.load(workspace["dir"] / "models" / "logistic_regression.pkl")
    offline_scores = model.predict_proba(scaler.transform(sample_df[flask_app.feature_columns]))[:, 1]
    offline_attacks = int((offline_scores >= 0.5).sum())

    client = flask_app.app.test_client()
    file_data = {"file": (io.BytesIO(sample_df.to_csv(index=False).encode()), "sample.csv"), "model": "logistic_regression"}
    page_text = client.post("/upload", data=file_data, content_type="multipart/form-data").get_data(as_text=True)
    card_values = re.findall(r'<div class="label">Predicted Attacks</div><div class="value">([\d,]+)</div>', page_text)
    assert int(card_values[0].replace(",", "")) == offline_attacks


### IT-04: upload -> validation -> scaling -> prediction -> results table
def test_it04_row_counts_through_upload(flask_app, workspace):
    sample_df = pd.read_csv(workspace["dir"] / "static" / "sample_upload.csv")
    first_feature = flask_app.feature_columns[0]
    sample_df[first_feature] = sample_df[first_feature].astype(object)
    sample_df.loc[[0, 1], first_feature] = "inf"  # 2 rows that should be skipped
    rows_sent = len(sample_df)

    client = flask_app.app.test_client()
    file_data = {"file": (io.BytesIO(sample_df.to_csv(index=False).encode()), "sample.csv"), "model": "random_forest"}
    page_text = client.post("/upload", data=file_data, content_type="multipart/form-data").get_data(as_text=True)
    classified = int(re.search(r'Rows Classified</div><div class="value">([\d,]+)', page_text).group(1).replace(",", ""))
    skipped = int(re.search(r'Rows Skipped</div><div class="value">([\d,]+)', page_text).group(1).replace(",", ""))
    table_rows = page_text.count("<td class=\"correct\">") + page_text.count("<td class=\"wrong\">")
    assert skipped == 2
    assert classified + skipped == rows_sent
    assert table_rows == classified  # sample is under the 500-row display limit


### IT-05: Evaluator threshold data -> Results Explorer
def test_it05_threshold_metrics_match_scores(workspace):
    models_dir = workspace["dir"] / "models"
    test_data = joblib.load(models_dir / "test_data.pkl")
    scores = joblib.load(models_dir / "test_scores.pkl")
    with open(models_dir / "results_explorer.json") as results_file:
        explorer = json.load(results_file)
    y_test = test_data["y_test"]
    for model_key, rows in explorer["threshold_metrics"].items():
        for row in rows:
            y_pred = (scores[model_key] >= row["threshold"]).astype(int)
            assert row["fp"] == int(((y_pred == 1) & (y_test == 0)).sum())
            assert row["fn"] == int(((y_pred == 0) & (y_test == 1)).sum())


def test_it05_results_page_receives_explorer_data(flask_app):
    page_text = flask_app.app.test_client().get("/results").get_data(as_text=True)
    first_row_id = str(flask_app.results_explorer_data["sample_rows"][0]["row_id"])
    assert first_row_id in page_text
