# backend/app/services/model_service.py
import joblib
import numpy as np
import os

MODEL_DIR = "./data/models"

# Load models
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

def predict(transaction):
    # Preprocess
    X = preprocessor.transform([transaction])

    # Individual predictions
    rf_proba = rf.predict_proba(X)[0, 1]
    xgb_proba = xgb.predict_proba(X)[0, 1]

    # Weighted ensemble (60% XGB + 40% RF)
    fraud_proba = 0.6 * xgb_proba + 0.4 * rf_proba

    # Risk level classification
    if fraud_proba >= 0.75:
        risk = "High"
    elif fraud_proba >= 0.25:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "fraud_probability": float(fraud_proba),
        "risk_level": risk
    }
