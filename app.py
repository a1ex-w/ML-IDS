### Imports
import json
import joblib
from flask import Flask, render_template, request, redirect, url_for

### Create Flask app
app = Flask(__name__)

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

### Page 2: Results Explorer (in progress)
@app.route("/results")
def results():
    return render_template("placeholder.html", page_title="Results Explorer")

### Page 3: About (in progress)
@app.route("/about")
def about():
    return render_template("placeholder.html", page_title="About")

### Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # 5001 because macOS AirPlay uses 5000