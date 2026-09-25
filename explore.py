"""Basic EDA for the CICIDS2017 NIDS dataset (MachineLearningCVE CSVs)."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/MachineLearningCVE")


def load_all() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        df["source_file"] = csv_path.name
        frames.append(df)
        print(f"loaded {csv_path.name}: {df.shape}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load_all()
    print("\n=== combined shape ===")
    print(df.shape)

    print("\n=== label counts ===")
    print(df["Label"].value_counts())

    print("\n=== missing values (top 10 columns) ===")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\n=== dtypes ===")
    print(df.dtypes.value_counts())

    print("\n=== numeric summary (first 5 cols) ===")
    numeric_cols = df.select_dtypes("number").columns[:5]
    print(df[numeric_cols].describe())


if __name__ == "__main__":
    main()
