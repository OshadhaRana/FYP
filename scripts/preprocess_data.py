import pandas as pd
import numpy as np
import os

RAW = "./data/raw/PS_20174392719_1491204439457_log.csv"
OUT = "./data/processed/paysim_processed.csv"

def main(sample_size=100_000, random_state=42):
    df = pd.read_csv(RAW)
    fraud = df[df['isFraud'] == 1]
    need_normals = max(0, sample_size - len(fraud))
    normals = df[df['isFraud'] == 0].sample(n=need_normals, random_state=random_state)
    sample = pd.concat([fraud, normals]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sample.to_csv(OUT, index=False)
    print(f"Saved sample to {OUT} (rows={len(sample)})")

if __name__ == "__main__":
    main()
