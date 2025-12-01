from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import shap
import joblib
import pandas as pd
import os

# Router setup
router = APIRouter()

# Load trained models
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
rf_model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))

# Request schema
class ExplainRequest(BaseModel):
    transaction: dict
    stakeholder: str  # "risk_analyst" or "compliance_officer"

@router.post("/explain")
def explain_transaction(request: ExplainRequest):
    try:
        # Convert transaction dict → dataframe
        df = pd.DataFrame([request.transaction])

        # Encode categorical feature if present
        if "type" in df.columns:
            df["type"] = df["type"].astype("category").cat.codes

        # Ensemble prediction
        rf_proba = rf_model.predict_proba(df)[:, 1]
        xgb_proba = xgb_model.predict_proba(df)[:, 1]
        fraud_proba = 0.4 * rf_proba + 0.6 * xgb_proba

        # SHAP explanation (using RF as example)
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(df)

        # Format explanation based on stakeholder
        if request.stakeholder == "risk_analyst":
            explanation = {
                "fraud_probability": float(fraud_proba[0]),
                "shap_values": shap_values[1].tolist(),
                "features": df.iloc[0].to_dict()
            }
        elif request.stakeholder == "compliance_officer":
            explanation = {
                "fraud_probability": float(fraud_proba[0]),
                "risk_level": "High" if fraud_proba[0] > 0.7 else "Medium" if fraud_proba[0] > 0.3 else "Low",
                "regulatory_notes": "This transaction requires AML review." if fraud_proba[0] > 0.7 else "No immediate AML concerns."
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid stakeholder type")

        return explanation

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
