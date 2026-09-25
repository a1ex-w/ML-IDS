### Imports
import json
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
from scipy.special import expit  # numerically safe sigmoid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

### Load test data and feature names
test_data = joblib.load("models/test_data.pkl")
X_test = test_data["X_test"]
y_test = test_data["y_test"]
feature_columns = joblib.load("models/feature_columns.pkl")

### Load trained models
models = {
    "random_forest": joblib.load("models/random_forest.pkl"),
    "logistic_regression": joblib.load("models/logistic_regression.pkl"),
    "linear_svm": joblib.load("models/linear_svm.pkl"),
}
display_names = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
}

os.makedirs("static/plots", exist_ok=True)
all_metrics = {}
all_scores = {}

### Evaluate each model
for model_key, model in models.items():
    model_name = display_names[model_key]
    y_pred = model.predict(X_test)

    # attack scores (0-1) used for ROC-AUC and the threshold slider
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        decision_values = model.decision_function(X_test)  # LinearSVC has no predict_proba
        y_score = expit(decision_values)  # sigmoid so 0.5 matches predict(); not a calibrated probability
    all_scores[model_key] = y_score

    # metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    model_metrics = {
        "name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_score), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    all_metrics[model_key] = model_metrics
    print(f"{model_name}: {model_metrics}")

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"])
    cm_display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"static/plots/cm_{model_key}.png", dpi=120, bbox_inches="tight")
    plt.close()

### Random Forest feature importance (top 15)
rf_importances = models["random_forest"].feature_importances_
top_indices = np.argsort(rf_importances)[-15:]  # 15 highest, ascending for barh
top_features = [feature_columns[i] for i in top_indices]
top_values = rf_importances[top_indices]

plt.figure(figsize=(8, 6))
plt.barh(top_features, top_values, color="steelblue")
plt.xlabel("Importance")
plt.title("Random Forest - Top 15 Feature Importances")
plt.savefig("static/plots/feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()

### Save metrics and scores
with open("models/metrics.json", "w") as metrics_file:
    json.dump(all_metrics, metrics_file, indent=2)
joblib.dump(all_scores, "models/test_scores.pkl")  # used by the threshold slider on Results Explorer

print("\nSaved metrics.json, test_scores.pkl, and plots to static/plots/")