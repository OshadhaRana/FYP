import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

# Import your data loader
from backend.app.utils.data_loader import load_data

def preprocess(df):
    df = df.copy()
    # Encode categorical columns automatically
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Separate features and target
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    return X, y

def main():
    # Load and preprocess
    print("Loading dataset...")
    df = load_data()
    print("Preprocessing dataset...")
    X, y = preprocess(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_path = os.path.join('backend', 'app', 'models', 'rf_model.pkl')
    os.makedirs(os.path.dirname(rf_path), exist_ok=True)
    joblib.dump(rf, rf_path)
    print("Random Forest saved.")

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
    xgb_path = os.path.join('backend', 'app', 'models', 'xgb_model.pkl')
    joblib.dump(xgb_model, xgb_path)
    print("XGBoost saved.")

    # Ensemble Predictions (average probabilities)
    print("Evaluating ensemble...")
    rf_probs = rf.predict_proba(X_test)[:,1]
    xgb_probs = xgb_model.predict_proba(X_test)[:,1]

    ensemble_probs = (rf_probs + xgb_probs) / 2
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, ensemble_preds)
    precision = precision_score(y_test, ensemble_preds)
    recall = recall_score(y_test, ensemble_preds)
    auc = roc_auc_score(y_test, ensemble_probs)

    print("\nEnsemble Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
