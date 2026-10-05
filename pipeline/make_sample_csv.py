### Imports
import pandas as pd

### Load cleaned dataset
cleaned_df = pd.read_csv("data/cleaned.csv")

### Take 100 benign rows plus up to 20 rows of each attack type
sample_parts = []
for attack_name, attack_group in cleaned_df.groupby("Attack Type"):
    rows_to_take = 100 if attack_name == "BENIGN" else min(20, len(attack_group))
    sample_parts.append(attack_group.sample(n=rows_to_take, random_state=42))
sample_df = pd.concat(sample_parts).sample(frac=1, random_state=42)  # shuffle so attack types are mixed

### Match the upload format: feature columns plus an optional Label column
sample_df = sample_df.rename(columns={"Attack Type": "Label"})
sample_df = sample_df.drop(columns=["is_attack"])

### Save where Flask can serve it for download
sample_df.to_csv("static/sample_upload.csv", index=False)
print(f"Saved static/sample_upload.csv: {len(sample_df)} rows, {sample_df.shape[1]} columns")