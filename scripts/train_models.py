import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

from backend.app.utils.data_loader import load_data

def preprocess(df):
    df = df.copy()

    # Identify categorical columns automatically
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Separate features and target
    X = df.drop(columns=['isFraud'])  # Replace with your target column
    y = df['isFraud']

    return X, y

def main():
    print("Loading dataset...")
    df = load_data()

    print("Preprocessing dataset...")
    X, y = preprocess(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensure models directory exists
    models_dir = os.path.join('backend', 'app', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(models_dir, 'rf_model.pkl'))
    print("Random Forest saved in backend/app/models/.")

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
    print("XGBoost saved in backend/app/models/.")

    print("Training complete!")

if __name__ == "__main__":
    main()
