from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib
import json

from backend.app.core.config import ALLOW_ORIGINS, MODEL_DIR
from backend.app.models.deep_learning_models import LSTMFraudDetector, CNN1DFraudDetector, MultiModelEnsemble
from backend.app.utils.preprocessing import TransactionPreprocessor
from backend.app.utils.sequence_preprocessing import SequencePreprocessor, TemporalFeatureEngineer
from backend.app.explainers.deep_learning_explainer import EnsembleExplainer

# ----------------------------
# FastAPI app setup
# ----------------------------
app = FastAPI(title="XAI-FinCrime API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Models
# ----------------------------
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

class ExplainRequest(BaseModel):
    transaction: Transaction
    stakeholder: str  # "risk_analyst" or "compliance_officer"

# ----------------------------
# Model Loading
# ----------------------------
fraud_model = None  # Multi-model ensemble
preprocessor = None
seq_preprocessor = None
temporal_engineer = None
ensemble_explainer = None
ensemble_config = None

@app.on_event("startup")
async def load_models():
    global fraud_model, preprocessor, seq_preprocessor, temporal_engineer, ensemble_explainer, ensemble_config

    try:
        # Load traditional models
        rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.joblib'))
        xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.joblib'))

        # Load deep learning models
        lstm_model = LSTMFraudDetector.load(os.path.join(MODEL_DIR, 'lstm_model'))
        cnn_model = CNN1DFraudDetector.load(os.path.join(MODEL_DIR, 'cnn_model'))

        # Load preprocessors
        preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
        seq_preprocessor = joblib.load(os.path.join(MODEL_DIR, 'sequence_preprocessor.joblib'))
        temporal_engineer = joblib.load(os.path.join(MODEL_DIR, 'temporal_engineer.joblib'))

        # Load ensemble config
        with open(os.path.join(MODEL_DIR, 'ensemble_config.json'), 'r') as f:
            ensemble_config = json.load(f)

        # Create multi-model ensemble
        fraud_model = MultiModelEnsemble(rf_model, xgb_model, lstm_model, cnn_model)
        fraud_model.weights = ensemble_config['weights']

        # Create ensemble explainer
        ensemble_explainer = EnsembleExplainer(rf_model, xgb_model, lstm_model, cnn_model, preprocessor)

        print("✓ All models loaded successfully!")

    except Exception as e:
        print(f"Warning: Could not load all models: {e}")
        print("Using fallback initialization...")
        # Fallback for POC
        preprocessor = TransactionPreprocessor()
        fraud_model = None

# ----------------------------
# Predict Endpoint
# ----------------------------
@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    if fraud_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([transaction.dict()])

    try:
        # Prepare tabular data
        X_tabular, _ = preprocessor.transform(df, fit=False)

        # Prepare sequence data (for single transaction, create a dummy sequence)
        # In production, you'd look up user's recent transactions
        df_temporal = temporal_engineer.transform(df.copy())
        feature_cols = ensemble_config['feature_columns'] + ensemble_config['temporal_features']
        X_seq, _ = seq_preprocessor.transform(df_temporal, feature_cols)

        # Get predictions from ensemble
        predictions_dict = fraud_model.predict_proba(X_tabular, X_seq)

        ensemble_proba = float(predictions_dict['ensemble'][0])

        return {
            "fraud_probability": ensemble_proba,
            "risk_level": "High" if ensemble_proba > 0.5 else "Medium" if ensemble_proba > 0.2 else "Low",
            "model_predictions": {
                "random_forest": float(predictions_dict['rf'][0]),
                "xgboost": float(predictions_dict['xgb'][0]),
                "lstm": float(predictions_dict['lstm'][0]),
                "cnn": float(predictions_dict['cnn'][0])
            },
            "ensemble_weights": fraud_model.weights
        }
    except Exception as e:
        # Fallback for when models aren't trained yet
        amt = float(df['amount'].iloc[0])
        proba = float(np.tanh(amt / 1e5) * 0.3)
        return {
            "fraud_probability": proba,
            "risk_level": "High" if proba > 0.5 else "Medium" if proba > 0.2 else "Low",
            "note": "Using fallback heuristic - models not trained yet"
        }

# ----------------------------
# Explain Endpoint
# ----------------------------
@app.post("/explain")
async def explain_transaction(request: ExplainRequest):
    if fraud_model is None or ensemble_explainer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([request.transaction.dict()])

    try:
        # Prepare tabular data
        X_tabular, _ = preprocessor.transform(df, fit=False)

        # Prepare sequence data
        df_temporal = temporal_engineer.transform(df.copy())
        feature_cols = ensemble_config['feature_columns'] + ensemble_config['temporal_features']
        X_seq, _ = seq_preprocessor.transform(df_temporal, feature_cols)

        # Get ensemble prediction
        predictions_dict = fraud_model.predict_proba(X_tabular, X_seq)
        ensemble_proba = float(predictions_dict['ensemble'][0])

        # Generate comprehensive explanation using EnsembleExplainer
        explanation = ensemble_explainer.explain_ensemble_prediction(
            X_tabular=X_tabular,
            X_seq=X_seq,
            ensemble_weights=fraud_model.weights,
            stakeholder_type=request.stakeholder,
            transaction_data=df
        )

        # Add prediction info
        explanation['fraud_probability'] = ensemble_proba
        explanation['model_predictions'] = {
            'random_forest': float(predictions_dict['rf'][0]),
            'xgboost': float(predictions_dict['xgb'][0]),
            'lstm': float(predictions_dict['lstm'][0]),
            'cnn': float(predictions_dict['cnn'][0])
        }

        return explanation

    except Exception as e:
        # Fallback explanation
        amt = float(df['amount'].iloc[0])
        proba = float(np.tanh(amt / 1e5) * 0.3)

        if request.stakeholder == "risk_analyst":
            return {
                "fraud_probability": proba,
                "model_confidence": round(proba * 100, 2),
                "feature_importance": {"amount": amt},
                "note": "Using fallback - models not trained yet"
            }
        elif request.stakeholder == "compliance_officer":
            return {
                "fraud_probability": proba,
                "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low",
                "regulatory_notes": "AML review required" if proba > 0.7 else "No immediate AML concerns",
                "note": "Using fallback - models not trained yet"
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid stakeholder type")

# ----------------------------
# Health Check
# ----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
