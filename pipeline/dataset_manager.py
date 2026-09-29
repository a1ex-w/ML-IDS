### Imports
import glob
import numpy as np
import pandas as pd

### Load and merge the 8 CICIDS2017 CSVs
csv_files = sorted(glob.glob("data/MachineLearningCVE/*.csv"))  # raw CICIDS2017 CSVs
print(f"Found {len(csv_files)} CSV files")

frames = []
for csv_path in csv_files:
    frame = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)  # latin-1 handles odd chars in web attack labels
    print(f"  {csv_path}: {len(frame):,} rows")
    frames.append(frame)

raw_df = pd.concat(frames, ignore_index=True)
print(f"Merged shape: {raw_df.shape}")

### Clean column headers
raw_df.columns = raw_df.columns.str.strip()  # CICIDS2017 headers have leading spaces

### Drop duplicate Fwd Header Length column
if "Fwd Header Length.1" in raw_df.columns:
    raw_df = raw_df.drop(columns=["Fwd Header Length.1"])
    print("Dropped duplicate column: Fwd Header Length.1")

### Replace inf values and drop rows with missing values
rows_before_na = len(raw_df)
raw_df = raw_df.replace([np.inf, -np.inf], np.nan)
raw_df = raw_df.dropna()
rows_dropped_na = rows_before_na - len(raw_df)
print(f"Dropped {rows_dropped_na:,} rows with inf/NaN values")

### Drop constant columns (same value in every row, no information)
constant_columns = [col for col in raw_df.columns if col != "Label" and raw_df[col].nunique() == 1]
raw_df = raw_df.drop(columns=constant_columns)
print(f"Dropped {len(constant_columns)} constant columns: {constant_columns}")

### Drop rows with invalid negative values
# Init_Win_bytes uses -1 to mean "no window observed", so those columns are skipped
negative_check_columns = [col for col in raw_df.columns if col not in ["Label", "Init_Win_bytes_forward", "Init_Win_bytes_backward"]]
negative_mask = (raw_df[negative_check_columns] < 0).any(axis=1)
raw_df = raw_df[~negative_mask]
print(f"Dropped {negative_mask.sum():,} rows with invalid negative values")

### Drop exact duplicate rows
rows_before_dupes = len(raw_df)
raw_df = raw_df.drop_duplicates()
rows_dropped_dupes = rows_before_dupes - len(raw_df)
print(f"Dropped {rows_dropped_dupes:,} duplicate rows")

### Clean up label text
raw_df["Label"] = raw_df["Label"].str.strip()
raw_df["Label"] = raw_df["Label"].str.replace(r"[^\x00-\x7F]+", "-", regex=True)  # fix garbled dash in "Web Attack - ..."

### Create binary and multiclass labels
raw_df["Attack Type"] = raw_df["Label"]  # multiclass label, e.g. DDoS, PortScan
raw_df["is_attack"] = (raw_df["Label"] != "BENIGN").astype(int)  # binary label: 0 = benign, 1 = attack
raw_df = raw_df.drop(columns=["Label"])

### Feature columns
feature_columns = [col for col in raw_df.columns if col not in ["Attack Type", "is_attack"]]

### Print class balance
print(f"\nFinal shape: {raw_df.shape}")
print(f"Feature count: {len(feature_columns)}")

print("\nBinary class balance:")
print(raw_df["is_attack"].value_counts())
print(raw_df["is_attack"].value_counts(normalize=True).round(4))

print("\nAttack type counts:")
print(raw_df["Attack Type"].value_counts())

### Save cleaned dataset
cleaned_path = "data/cleaned.csv"
raw_df.to_csv(cleaned_path, index=False)
print(f"\nSaved cleaned dataset to {cleaned_path}")