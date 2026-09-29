### Imports
import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Load cleaned dataset
cleaned_df = pd.read_csv("data/cleaned.csv")
print(f"Loaded cleaned dataset: {cleaned_df.shape}")

### Take a stratified sample to keep training time reasonable
sample_fraction = 0.2
sampled_df, _ = train_test_split(
    cleaned_df,
    train_size=sample_fraction,
    stratify=cleaned_df["Attack Type"],  # keeps every attack type in the sample
    random_state=42,
)
print(f"Sampled {len(sampled_df):,} rows ({sample_fraction:.0%})")

### Separate features and labels
feature_columns = [col for col in sampled_df.columns if col not in ["Attack Type", "is_attack"]]
X = sampled_df[feature_columns]
y = sampled_df["is_attack"]
attack_types = sampled_df["Attack Type"]

### Train/test split (80/20)
X_train, X_test, y_train, y_test, attack_train, attack_test = train_test_split(
    X, y, attack_types,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

### Scale features (fit on training data only to avoid leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Train Random Forest
print("\nTraining Random Forest...")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_train_time = time.time() - start_time
rf_pred = rf_model.predict(X_test_scaled)
print(f"  Time: {rf_train_time:.1f}s")
print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f} | Precision: {precision_score(y_test, rf_pred):.4f} | Recall: {recall_score(y_test, rf_pred):.4f} | F1: {f1_score(y_test, rf_pred):.4f}")

### Train Logistic Regression
print("\nTraining Logistic Regression...")
start_time = time.time()
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_train_time = time.time() - start_time
lr_pred = lr_model.predict(X_test_scaled)
print(f"  Time: {lr_train_time:.1f}s")
print(f"  Accuracy: {accuracy_score(y_test, lr_pred):.4f} | Precision: {precision_score(y_test, lr_pred):.4f} | Recall: {recall_score(y_test, lr_pred):.4f} | F1: {f1_score(y_test, lr_pred):.4f}")

### Train Linear SVM
print("\nTraining Linear SVM...")
start_time = time.time()
svm_model = LinearSVC(class_weight="balanced", dual=False, max_iter=2000, random_state=42)  # LinearSVC scales to large data, regular SVC does not
svm_model.fit(X_train_scaled, y_train)
svm_train_time = time.time() - start_time
svm_pred = svm_model.predict(X_test_scaled)
print(f"  Time: {svm_train_time:.1f}s")
print(f"  Accuracy: {accuracy_score(y_test, svm_pred):.4f} | Precision: {precision_score(y_test, svm_pred):.4f} | Recall: {recall_score(y_test, svm_pred):.4f} | F1: {f1_score(y_test, svm_pred):.4f}")

### Save models, scaler, and test data
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(lr_model, "models/logistic_regression.pkl")
joblib.dump(svm_model, "models/linear_svm.pkl")
joblib.dump(scaler, "models/scaler.pkl")  # needed later to scale uploaded CSVs
joblib.dump(feature_columns, "models/feature_columns.pkl")  # expected upload schema
joblib.dump(
    {"X_test": X_test_scaled, "y_test": y_test.values, "attack_test": attack_test.values},
    "models/test_data.pkl",  # used by the Evaluator and dashboard
)
print("\nSaved models, scaler, feature list, and test data to models/")