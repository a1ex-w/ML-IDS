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

### Results Explorer: thresholds to test (0.05 to 0.95)
attack_test = test_data["attack_test"]
thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

### Results Explorer: metrics at each threshold (for the slider)
threshold_metrics = {}
for model_key, y_score in all_scores.items():
    model_rows = []
    for threshold in thresholds:
        y_pred_t = (y_score >= threshold).astype(int)  # flag as attack if score is at or above threshold
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        model_rows.append({
            "threshold": threshold,
            "precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_t, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_t, zero_division=0), 4),
            "fp": int(fp),
            "fn": int(fn),
        })
    threshold_metrics[model_key] = model_rows

### Results Explorer: correct rate per attack type at each threshold (for the filter)
attack_metrics = {}
for model_key, y_score in all_scores.items():
    model_attacks = {}
    for attack_name in np.unique(attack_test):
        attack_mask = attack_test == attack_name
        expected_label = 0 if attack_name == "BENIGN" else 1  # benign should be 0, every attack should be 1
        correct_rates = []
        for threshold in thresholds:
            preds_for_type = (y_score[attack_mask] >= threshold).astype(int)
            correct_rates.append(round(float((preds_for_type == expected_label).mean()), 4))
        model_attacks[str(attack_name)] = {"count": int(attack_mask.sum()), "correct_rates": correct_rates}
    attack_metrics[model_key] = model_attacks

### Results Explorer: sample rows for the prediction table (up to 25 per attack type)
rng = np.random.default_rng(42)
sample_rows = []
for attack_name in np.unique(attack_test):
    type_indices = np.where(attack_test == attack_name)[0]
    picked = rng.choice(type_indices, size=min(25, len(type_indices)), replace=False)
    for row_index in picked:
        sample_rows.append({
            "row_id": int(row_index),
            "attack_type": str(attack_name),
            "true_label": int(y_test[row_index]),
            "random_forest": round(float(all_scores["random_forest"][row_index]), 4),
            "logistic_regression": round(float(all_scores["logistic_regression"][row_index]), 4),
            "linear_svm": round(float(all_scores["linear_svm"][row_index]), 4),
        })

### Save Results Explorer data
results_explorer_data = {
    "thresholds": thresholds,
    "threshold_metrics": threshold_metrics,
    "attack_metrics": attack_metrics,
    "sample_rows": sample_rows,
}
with open("models/results_explorer.json", "w") as results_file:
    json.dump(results_explorer_data, results_file)
print(f"Saved results_explorer.json ({len(thresholds)} thresholds, {len(sample_rows)} sample rows)")