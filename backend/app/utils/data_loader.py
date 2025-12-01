import os
import pandas as pd

def load_data():
    # Go 4 levels up from backend/app/utils/data_loader.py
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    file_path = os.path.join(base_dir, "data", "processed", "paysim_processed.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed dataset not found at {file_path}")

    return pd.read_csv(file_path)
