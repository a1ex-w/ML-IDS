### Imports
import numpy as np
import pandas as pd
import json
import joblib
from flask import Flask, render_template, request, redirect, url_for
from scipy.special import expit  # sigmoid for Linear SVM scores, same as evaluator.py

### Create Flask app
app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # reject uploads over 50 MB

### Load models and evaluation results at startup
with open("models/metrics.json") as metrics_file:
    all_metrics = json.load(metrics_file)

loaded_models = {
    "random_forest": joblib.load("models/random_forest.pkl"),
    "logistic_regression": joblib.load("models/logistic_regression.pkl"),
    "linear_svm": joblib.load("models/linear_svm.pkl"),
}
scaler = joblib.load("models/scaler.pkl")  # used for uploaded CSVs later
feature_columns = joblib.load("models/feature_columns.pkl")
with open("models/results_explorer.json") as results_file:
    results_explorer_data = json.load(results_file) # threshold metrics, per-attack rates, sample rows
print("Models and metrics loaded")

### Home redirects to the dashboard
@app.route("/")
def index():
    return redirect(url_for("dashboard"))

### Page 1: Model Evaluation
@app.route("/dashboard")
def dashboard():
    selected_model = request.args.get("model", "random_forest")
    if selected_model not in all_metrics:
        selected_model = "random_forest"  # fall back if the query string is invalid
    selected_metrics = all_metrics[selected_model]
    return render_template(
        "dashboard.html",
        all_metrics=all_metrics,
        selected_model=selected_model,
        selected_metrics=selected_metrics,
    )

### Page 2: Results Explorer
@app.route("/results")
def results():
    model_names = {model_key: metrics["name"] for model_key, metrics in all_metrics.items()}
    return render_template(
        "results.html",
        results_data=results_explorer_data,
        model_names=model_names,
        feature_count=len(feature_columns),
    )

### Upload: classify a user CSV
@app.route("/upload", methods=["POST"])
def upload():
    selected_model = request.form.get("model", "random_forest")
    if selected_model not in loaded_models:
        selected_model = "random_forest"  # fall back if the form value is invalid
    uploaded_file = request.files.get("file")

    # check that a .csv file was chosen
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("upload_results.html", error="No file was selected.")
    if not uploaded_file.filename.lower().endswith(".csv"):
        return render_template("upload_results.html", error="Only .csv files are accepted.")

    # read the CSV
    try:
        upload_df = pd.read_csv(uploaded_file, encoding="latin-1", low_memory=False)
    except Exception:
        return render_template("upload_results.html", error="The file could not be read as a CSV.")
    upload_df.columns = upload_df.columns.str.strip()  # raw CICFlowMeter headers have leading spaces

    # schema check: every feature the models were trained on must be present
    missing_columns = [col for col in feature_columns if col not in upload_df.columns]
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:5])
        error_message = f"The file is missing {len(missing_columns)} required columns (for example: {missing_preview}). See the About page for the required format."
        return render_template("upload_results.html", error=error_message)

    # keep only model features, force numbers, and skip rows with bad values
    features_df = upload_df[feature_columns].apply(pd.to_numeric, errors="coerce")  # text becomes NaN
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    valid_mask = features_df.notna().all(axis=1)
    rows_skipped = int((~valid_mask).sum())
    features_df = features_df[valid_mask]
    if len(features_df) == 0:
        return render_template("upload_results.html", error="No valid rows were found. Check that the feature columns contain numbers.")

    # scale with the training scaler, then predict at the default 0.50 threshold
    X_upload = scaler.transform(features_df)
    model = loaded_models[selected_model]
    if hasattr(model, "predict_proba"):
        upload_scores = model.predict_proba(X_upload)[:, 1]
    else:
        upload_scores = expit(model.decision_function(X_upload))  # LinearSVC has no predict_proba
    upload_preds = (upload_scores >= 0.5).astype(int)

    # if the file has a Label column, compare predictions to it
    true_labels = None
    label_text = None
    accuracy = None
    if "Label" in upload_df.columns:
        label_text = upload_df.loc[valid_mask, "Label"].astype(str).str.strip().values
        true_labels = (label_text != "BENIGN").astype(int)
        accuracy = round(float((true_labels == upload_preds).mean()) * 100, 2)

    # build table rows (first 500 only, to keep the page fast)
    file_row_numbers = upload_df.index[valid_mask] + 2  # +2 so numbers match spreadsheet rows (header is row 1)
    table_rows = []
    for i in range(min(500, len(upload_preds))):
        table_rows.append({
            "file_row": int(file_row_numbers[i]),
            "predicted": "Attack" if upload_preds[i] == 1 else "Benign",
            "score": round(float(upload_scores[i]), 4),
            "label": label_text[i] if label_text is not None else None,
            "correct": bool(true_labels[i] == upload_preds[i]) if true_labels is not None else None,
        })

    # summary numbers for the cards
    summary = {
        "rows_classified": len(upload_preds),
        "predicted_attacks": int(upload_preds.sum()),
        "predicted_benign": int(len(upload_preds) - upload_preds.sum()),
        "rows_skipped": rows_skipped,
        "accuracy": accuracy,
    }
    return render_template(
        "upload_results.html",
        model_name=all_metrics[selected_model]["name"],
        file_name=uploaded_file.filename,
        summary=summary,
        table_rows=table_rows,
    )

### Page 3: About
@app.route("/about")
def about():
    return render_template("about.html", feature_columns=feature_columns)

### Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # 5001 because macOS AirPlay uses 5000