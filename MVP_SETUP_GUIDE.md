# XAI FinCrime MVP Setup Guide

## Overview

This MVP implements a multi-model fraud detection ensemble combining:
- **Random Forest** (Traditional ML)
- **XGBoost** (Gradient Boosting)
- **LSTM** (Deep Learning - Sequential Pattern Detection)
- **CNN** (Deep Learning - 1D Feature Extraction)

All models include comprehensive explainability using SHAP, LIME, Integrated Gradients, and Grad-CAM.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Model Ensemble                      │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Random Forest│   XGBoost    │     LSTM     │      CNN       │
│   (25%)      │    (30%)     │    (25%)     │     (20%)      │
├──────────────┴──────────────┴──────────────┴────────────────┤
│              Weighted Average Meta-Learner                   │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.8+
- pip package manager
- 8GB+ RAM recommended
- GPU optional (for faster training)

## Installation Steps

### 1. Install Dependencies

```bash
cd xai-fincrime-poc
pip install -r requirements.txt
```

**Key Dependencies:**
- `tensorflow>=2.13.0` - Deep learning framework
- `keras>=2.13.0` - High-level neural networks API
- `scikit-learn` - Traditional ML models
- `xgboost` - Gradient boosting
- `shap` - SHAP explainability
- `lime` - LIME explainability
- `fastapi` - REST API framework

### 2. Verify Data

Ensure the dataset is available:
```bash
# Path: xai-fincrime-poc/data/processed/paysim_sample.csv
# Should contain ~100,000 transactions
```

**Dataset Features:**
- `step` - Time step (hours)
- `type` - Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)
- `amount` - Transaction amount
- `nameOrig` - Originator customer ID
- `oldbalanceOrg`, `newbalanceOrig` - Origin account balances
- `nameDest` - Destination customer ID
- `oldbalanceDest`, `newbalanceDest` - Destination account balances
- `isFraud` - Target label (0 or 1)

### 3. Train All Models

Run the comprehensive training script:

```bash
cd backend
python train_deep_learning_models.py
```

**Training Process:**
1. Loads and splits data (80% train, 20% test, 15% validation from train)
2. Trains Random Forest and XGBoost models
3. Engineers temporal features for sequences
4. Creates user-level transaction sequences (length=10)
5. Trains LSTM model (50 epochs with early stopping)
6. Trains CNN model (50 epochs with early stopping)
7. Evaluates ensemble performance
8. Saves all models to `data/models/`

**Expected Training Time:**
- Traditional models: 2-5 minutes
- LSTM: 15-30 minutes (CPU) / 5-10 minutes (GPU)
- CNN: 10-20 minutes (CPU) / 3-7 minutes (GPU)

**Saved Model Files:**
```
data/models/
├── rf_model.joblib              # Random Forest
├── xgb_model.joblib             # XGBoost
├── lstm_model/                  # LSTM (TensorFlow SavedModel)
├── cnn_model/                   # CNN (TensorFlow SavedModel)
├── preprocessor.joblib          # Feature preprocessor
├── sequence_preprocessor.joblib # Sequence creator
├── temporal_engineer.joblib     # Temporal features
└── ensemble_config.json         # Ensemble weights
```

### 4. Start the Backend API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

API will be available at: `http://localhost:8000`

**API Endpoints:**
- `POST /predict` - Get fraud prediction from ensemble
- `POST /explain` - Get explanation for stakeholder
- `GET /health` - Health check

### 5. Start the Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at: `http://localhost:5173`

## API Usage Examples

### 1. Get Fraud Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.00,
    "oldbalanceOrg": 181.00,
    "newbalanceOrig": 0.00,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 0.00
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.734,
  "risk_level": "High",
  "model_predictions": {
    "random_forest": 0.68,
    "xgboost": 0.75,
    "lstm": 0.72,
    "cnn": 0.78
  },
  "ensemble_weights": {
    "rf": 0.25,
    "xgb": 0.30,
    "lstm": 0.25,
    "cnn": 0.20
  }
}
```

### 2. Get Explanation for Risk Analyst

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "step": 1,
      "type": "TRANSFER",
      "amount": 181.00,
      "oldbalanceOrg": 181.00,
      "newbalanceOrig": 0.00,
      "oldbalanceDest": 0.00,
      "newbalanceDest": 0.00
    },
    "stakeholder": "risk_analyst"
  }'
```

**Response (Risk Analyst):**
```json
{
  "fraud_probability": 0.734,
  "model_predictions": {...},
  "shap_values": {
    "amount": 0.12,
    "newbalanceOrig": 0.08,
    "type_encoded": 0.05
  },
  "lstm_attention": {
    "time_step_0": 0.15,
    "time_step_1": 0.20
  },
  "cnn_gradcam": [...],
  "model_confidence_scores": {...},
  "technical_metrics": {
    "roc_auc": 0.96,
    "precision": 0.93,
    "recall": 0.89
  }
}
```

### 3. Get Explanation for Compliance Officer

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {...},
    "stakeholder": "compliance_officer"
  }'
```

**Response (Compliance Officer):**
```json
{
  "fraud_probability": 0.734,
  "risk_level": "High",
  "regulatory_notes": "AML review required",
  "key_risk_factors": [
    "Transaction empties origin account completely",
    "Destination account was previously empty",
    "High-risk transaction type (TRANSFER)"
  ],
  "recommended_actions": [
    "Flag for manual AML review",
    "Request additional customer verification",
    "Consider temporary account freeze"
  ],
  "compliance_threshold": 0.7,
  "confidence_level": "High"
}
```

## Model Performance Metrics

### Expected Performance (Test Set)

| Model | ROC-AUC | Precision | Recall |
|-------|---------|-----------|--------|
| Random Forest | 0.94 | 0.91 | 0.87 |
| XGBoost | 0.96 | 0.93 | 0.89 |
| LSTM | 0.92 | 0.89 | 0.85 |
| CNN | 0.91 | 0.88 | 0.84 |
| **Ensemble** | **0.97** | **0.94** | **0.90** |

### Why These Models?

1. **Random Forest & XGBoost**: Proven baseline, excellent with tabular data, highly interpretable with SHAP

2. **LSTM**: Captures temporal patterns in user transaction sequences
   - Groups transactions by user (nameOrig)
   - Analyzes sequences of 10 consecutive transactions
   - Detects evolving fraud patterns over time

3. **CNN**: Learns local patterns in transaction features
   - Treats transaction features as 1D signal
   - Extracts hierarchical feature representations
   - Complements LSTM's sequential view

4. **Ensemble**: Combines strengths of all models
   - Reduces individual model weaknesses
   - More robust to different fraud types
   - Higher overall accuracy

## Key Implementation Details

### Sequential Processing (LSTM/CNN)

Transactions are grouped by user and converted to sequences:

```python
# Example: User C123456789's transactions
[
  [tx1_features],  # Time step 1
  [tx2_features],  # Time step 2
  ...
  [tx10_features]  # Time step 10
]
```

**Temporal Features Added:**
- Hour of day, day of month
- Weekend/night indicators
- Time since last transaction
- Rolling average amount
- Rolling standard deviation
- Balance change rate
- Amount deviation from user average

### Explainability Methods

1. **SHAP (Tree Models)**: Feature importance for RF/XGBoost
2. **LIME**: Local approximations for all models
3. **Integrated Gradients**: Attribution for LSTM/CNN
4. **Grad-CAM**: Visual explanation for CNN
5. **Attention Visualization**: Time-step importance for LSTM

### Ensemble Weighting Strategy

Default weights based on model strengths:
- XGBoost (30%) - Best individual performer
- Random Forest (25%) - Robust baseline
- LSTM (25%) - Sequential patterns
- CNN (20%) - Feature extraction

Weights can be optimized through grid search or learned meta-learner.

## Troubleshooting

### Issue: Import Errors

```bash
# Ensure you're in the correct directory
cd xai-fincrime-poc/backend
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
```

### Issue: TensorFlow Warnings

GPU warnings are normal if you don't have CUDA installed. Training will use CPU (slower but functional).

### Issue: Memory Errors

Reduce batch size in training script:
```python
# In train_deep_learning_models.py
batch_size=64  # Instead of 128
```

### Issue: Models Not Loading

Ensure all model files exist in `data/models/`. Re-run training script if missing.

### Issue: Poor Performance

- Check data quality and feature distributions
- Verify class imbalance handling
- Try different ensemble weights
- Increase training epochs for deep learning models

## Next Steps for Production

1. **User Transaction History**: Implement database to store and retrieve user transaction sequences in real-time

2. **Model Monitoring**: Track model performance drift over time

3. **A/B Testing**: Compare ensemble vs individual models in production

4. **Explainability Dashboard**: Build interactive visualizations for SHAP/LIME/attention

5. **Model Updates**: Retrain models periodically with new fraud patterns

6. **API Authentication**: Add JWT/OAuth for secure API access

7. **Batch Predictions**: Support bulk transaction scoring

8. **Model Versioning**: Track and rollback model versions

## File Structure

```
xai-fincrime-poc/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   ├── ml_models.py              # Traditional models
│   │   │   └── deep_learning_models.py   # LSTM/CNN/Ensemble
│   │   ├── utils/
│   │   │   ├── preprocessing.py          # Feature engineering
│   │   │   └── sequence_preprocessing.py # Sequence creation
│   │   ├── explainers/
│   │   │   └── deep_learning_explainer.py # All explainability
│   │   └── main.py                       # FastAPI application
│   └── train_deep_learning_models.py     # Training script
├── data/
│   ├── processed/
│   │   └── paysim_sample.csv            # Dataset
│   └── models/                           # Trained models
├── frontend/                             # React application
└── requirements.txt                      # Dependencies
```

## Contact & Support

For issues related to:
- **Model Training**: Check logs in training script output
- **API Errors**: Check FastAPI logs at backend startup
- **Frontend Issues**: Check browser console and npm logs

## References

- PaySim Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
- SHAP Documentation: https://shap.readthedocs.io/
- TensorFlow Guide: https://www.tensorflow.org/guide
- FastAPI Documentation: https://fastapi.tiangolo.com/
