### Imports
import joblib
import numpy as np
import pandas as pd

### Load cleaned dataset
cleaned_df = pd.read_csv("data/cleaned.csv")
print(f"Shape: {cleaned_df.shape}")

### 1. Column headers
print("\n=== Columns ===")
for index, col in enumerate(cleaned_df.columns):
    print(f"{index:2d}  {col!r}")  # repr shows any hidden whitespace
duplicate_names = cleaned_df.columns[cleaned_df.columns.duplicated()].tolist()
whitespace_names = [col for col in cleaned_df.columns if col != col.strip()]
print(f"Duplicate column names: {duplicate_names}")
print(f"Names with leading/trailing spaces: {whitespace_names}")

### 2. Data types
print("\n=== Data types ===")
print(cleaned_df.dtypes.value_counts())
non_numeric_columns = cleaned_df.select_dtypes(exclude="number").columns.tolist()
print(f"Non-numeric columns: {non_numeric_columns}")  # should only be Attack Type

### 3. Missing and infinite values
feature_df = cleaned_df.drop(columns=["Attack Type", "is_attack"])
print("\n=== Missing / infinite ===")
print(f"NaN count: {feature_df.isna().sum().sum()}")
print(f"Inf count: {np.isinf(feature_df.values).sum()}")

### 4. Duplicate rows
print(f"Duplicate rows: {cleaned_df.duplicated().sum()}")

### 5. Label encoding check
print("\n=== Label encoding ===")
label_crosstab = pd.crosstab(cleaned_df["Attack Type"], cleaned_df["is_attack"])
print(label_crosstab)  # BENIGN should only be under 0, every attack only under 1

### 6. Constant columns (same value in every row)
constant_columns = [col for col in feature_df.columns if feature_df[col].nunique() == 1]
print(f"\nConstant columns ({len(constant_columns)}): {constant_columns}")

### 7. Negative values (counts, lengths, and durations shouldn't be negative)
negative_counts = (feature_df < 0).sum()
negative_counts = negative_counts[negative_counts > 0]
print("\n=== Columns with negative values ===")
print(negative_counts)

### 8. Value ranges for a few key columns
print("\n=== Key column ranges ===")
key_columns = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Flow Bytes/s"]
print(feature_df[key_columns].describe().T[["min", "max", "mean"]])

### 9. Saved feature list matches the data and excludes labels
feature_columns = joblib.load("models/feature_columns.pkl")
label_leaks = [col for col in feature_columns if col in ["Attack Type", "is_attack", "Label"]]
print("\n=== Model feature list ===")
print(f"Features used for training: {len(feature_columns)}")
print(f"Label columns in feature list: {label_leaks}")  # should be []
print(f"Matches cleaned feature columns: {feature_columns == feature_df.columns.tolist()}")