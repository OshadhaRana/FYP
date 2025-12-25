"""
Fraud Detection API
Human-Centric Explainable AI for Financial Crime Detection

Endpoints:
- POST /predict: Get fraud prediction with explanation
- GET /models: Get model information
- POST /explain: Get detailed explanation for transaction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import sys
import os

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simplified_ensemble import SimplifiedEnsemble, create_ensemble

# Initialize FastAPI app
app = FastAPI(
    title="Explainable Fraud Detection API",
    description="Human-Centric AI for Financial Crime Detection with SHAP and GNN explanations",
    version="2.0"
)

# Initialize ensemble (singleton)
ensemble = None


def get_ensemble():
    """Get or create ensemble instance"""
    global ensemble
    if ensemble is None:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            'data', 'models'
        )
        ensemble = create_ensemble(models_dir)
    return ensemble


# Request/Response Models
class TransactionRequest(BaseModel):
    """Transaction input for prediction"""
    step: int = Field(..., description="Time step (hour in simulation)")
    type: str = Field(..., description="Transaction type: TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT")
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    nameOrig: str = Field(..., description="Origin account ID")
    nameDest: str = Field(..., description="Destination account ID")
    oldbalanceOrg: float = Field(..., ge=0, description="Origin account balance BEFORE transaction")
    oldbalanceDest: float = Field(..., ge=0, description="Destination account balance BEFORE transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "step": 145,
                "type": "TRANSFER",
                "amount": 566142.89,
                "nameOrig": "C1357772203",
                "nameDest": "C585432949",
                "oldbalanceOrg": 566142.89,
                "oldbalanceDest": 0.0
            }
        }


class PredictionResponse(BaseModel):
    """Fraud prediction response"""
    fraud_probability: float
    prediction: str
    confidence: str
    recommended_action: str
    model_contributions: Dict[str, float]
    risk_factors: List[str]


class ExplanationResponse(BaseModel):
    """Detailed explanation response"""
    summary: str
    confidence: str
    model_breakdown: Dict[str, str]
    risk_factors: List[str]
    recommended_action: str
    transaction_details: Dict[str, str]


class ModelInfoResponse(BaseModel):
    """Model information response"""
    ensemble_name: str
    version: str
    models_loaded: List[str]
    weights: Dict[str, float]
    research_focus: str
    model_roles: Dict[str, str]


# API Endpoints
@app.get("/")
async def root():
    """API root - welcome message"""
    return {
        "message": "Explainable Fraud Detection API",
        "version": "2.0",
        "research_title": "Human-Centric Explainable AI for Financial Crime Detection",
        "endpoints": {
            "/predict": "POST - Get fraud prediction with explanation",
            "/explain": "POST - Get detailed explanation",
            "/models": "GET - Get model information",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ens = get_ensemble()
    return {
        "status": "healthy",
        "models_loaded": len(ens.models),
        "model_names": list(ens.models.keys())
    }


@app.get("/models", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the ensemble models"""
    ens = get_ensemble()
    return ens.get_model_info()


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud probability for a transaction

    Returns:
    - fraud_probability: Ensemble fraud probability (0-1)
    - prediction: "FRAUD" or "NORMAL"
    - confidence: "HIGH", "MEDIUM", or "LOW"
    - recommended_action: Action for fraud investigator
    - model_contributions: Individual model predictions
    - risk_factors: Identified risk factors
    """
    ens = get_ensemble()

    try:
        # Convert to dict
        txn_dict = transaction.model_dump()

        # Get prediction
        result = ens.predict_with_explanation(txn_dict)

        return PredictionResponse(
            fraud_probability=result.fraud_probability,
            prediction=result.prediction,
            confidence=result.confidence,
            recommended_action=result.recommended_action,
            model_contributions=result.model_contributions,
            risk_factors=result.risk_factors
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(transaction: TransactionRequest):
    """
    Get detailed human-readable explanation for a transaction

    Returns:
    - summary: Brief prediction summary
    - confidence: Confidence level
    - model_breakdown: Each model's contribution
    - risk_factors: Identified risk factors
    - recommended_action: Suggested next steps
    - transaction_details: Transaction information
    """
    ens = get_ensemble()

    try:
        # Convert to dict
        txn_dict = transaction.model_dump()

        # Get explanation
        explanation = ens.explain_prediction(txn_dict)

        return ExplanationResponse(**explanation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(transactions: List[TransactionRequest]):
    """
    Predict fraud for multiple transactions

    Returns list of predictions with explanations
    """
    ens = get_ensemble()
    results = []

    for i, txn in enumerate(transactions):
        try:
            txn_dict = txn.model_dump()
            result = ens.predict_with_explanation(txn_dict, transaction_id=i)
            results.append({
                "transaction_id": i,
                "fraud_probability": result.fraud_probability,
                "prediction": result.prediction,
                "confidence": result.confidence,
                "risk_factors": result.risk_factors
            })
        except Exception as e:
            results.append({
                "transaction_id": i,
                "error": str(e)
            })

    return {"predictions": results, "total": len(results)}


# Run with: uvicorn prediction_api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
