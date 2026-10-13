### TC-05 / TC-06: DashboardController (app.py)
# Uses Flask's test client, so no browser or running server is needed.

### Imports
import io

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def client(flask_app):
    return flask_app.app.test_client()


@pytest.fixture(scope="module")
def sample_csv_text(workspace):
    return (workspace["dir"] / "static" / "sample_upload.csv").read_text()


### TC-05: page routes
def test_tc05_home_redirects_to_dashboard(client):
    response = client.get("/")
    assert response.status_code == 302
    assert "/dashboard" in response.headers["Location"]


@pytest.mark.parametrize("model_key", ["random_forest", "logistic_regression", "linear_svm"])
def test_tc05_dashboard_each_model(client, flask_app, model_key):
    response = client.get(f"/dashboard?model={model_key}")
    assert response.status_code == 200
    assert flask_app.all_metrics[model_key]["name"] in response.get_data(as_text=True)


def test_tc05_dashboard_bad_model_falls_back(client):
    response = client.get("/dashboard?model=not_a_model")
    assert response.status_code == 200
    assert "Random Forest" in response.get_data(as_text=True)


def test_tc05_results_page(client):
    response = client.get("/results")
    assert response.status_code == 200


def test_tc05_about_page_lists_features(client, flask_app):
    response = client.get("/about")
    page_text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert flask_app.feature_columns[0] in page_text


def test_tc05_unknown_route_is_404(client):
    assert client.get("/does-not-exist").status_code == 404


### TC-06: CSV upload
def upload(client, file_text, file_name="test.csv", model_key="random_forest"):
    file_data = {"file": (io.BytesIO(file_text.encode("latin-1")), file_name), "model": model_key}
    return client.post("/upload", data=file_data, content_type="multipart/form-data")


@pytest.mark.parametrize("model_key", ["random_forest", "logistic_regression", "linear_svm"])
def test_tc06_valid_upload(client, sample_csv_text, model_key):
    response = upload(client, sample_csv_text, model_key=model_key)
    page_text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "error-box" not in page_text
    assert "Rows Classified" in page_text


def test_tc06_wrong_file_type_rejected(client):
    response = upload(client, "hello", file_name="notes.txt")
    assert "Only .csv files are accepted." in response.get_data(as_text=True)


def test_tc06_no_file_rejected(client):
    response = client.post("/upload", data={"model": "random_forest"}, content_type="multipart/form-data")
    assert "No file was selected." in response.get_data(as_text=True)


def test_tc06_missing_columns_rejected(client):
    response = upload(client, "Flow Duration,Label\n100,BENIGN\n")
    assert "missing" in response.get_data(as_text=True)
    assert "required columns" in response.get_data(as_text=True)


def test_tc06_empty_file_rejected(client):
    response = upload(client, "")
    assert "could not be read as a CSV" in response.get_data(as_text=True)


def test_tc06_bad_values_skipped(client, flask_app, sample_csv_text):
    sample_df = pd.read_csv(io.StringIO(sample_csv_text))
    first_feature = flask_app.feature_columns[0]
    sample_df[first_feature] = sample_df[first_feature].astype(object)
    sample_df.loc[0, first_feature] = "inf"
    sample_df.loc[1, first_feature] = None
    sample_df.loc[2, first_feature] = "abc"
    response = upload(client, sample_df.to_csv(index=False))
    page_text = response.get_data(as_text=True)
    assert "error-box" not in page_text
    assert "Rows skipped had missing, infinite, or non-numeric feature values." in page_text


def test_tc06_all_bad_rows_rejected(client, flask_app):
    header = ",".join(flask_app.feature_columns)
    bad_row = ",".join(["abc"] * len(flask_app.feature_columns))
    response = upload(client, f"{header}\n{bad_row}\n")
    assert "No valid rows were found." in response.get_data(as_text=True)


def test_tc06_oversized_upload_rejected(client):
    big_text = "a" * (51 * 1024 * 1024)  # over the 50 MB limit
    assert upload(client, big_text).status_code == 413


def test_tc06_sample_csv_downloads(client):
    response = client.get("/static/sample_upload.csv")
    assert response.status_code == 200
    response.close()
