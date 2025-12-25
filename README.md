# XAI FinCrime Detection System

Final Year Project - Explainable AI for Financial Fraud Detection

## About This Project

This is my final year project exploring how machine learning can be used to detect fraudulent financial transactions while providing explanations for the predictions. The main goal is to address the "black box" problem in fraud detection - where ML models make predictions but can't explain why.

The system uses multiple models working together (ensemble approach) and provides SHAP-based explanations so that fraud investigators can understand why a transaction was flagged.

## What It Does

- Detects potentially fraudulent transactions using a 3-model ensemble
- Provides feature importance explanations using SHAP
- Includes a dashboard for viewing predictions and explanations
- REST API for integrating with other systems

## Models Used

| Model | Weight | Why I Used It |
|-------|--------|---------------|
| XGBoost | 50% | Best performing model, works well with tabular data |
| Random Forest | 15% | Good baseline, validates XGBoost results |
| Graph Neural Network | 35% | Captures relationships between accounts |

Note: I originally tried 5 models (including CNN, TabTransformer, LSTM) but simplified to 3 for clearer research contribution. The archived models are in `archive/old_ensemble/`.

## Results

The ensemble achieves approximately 98.5% ROC-AUC on the test set. Individual model performance:

- XGBoost: 99.99% AUC
- Random Forest: 99.98% AUC  
- GNN: 94.36% AUC

The high accuracy is due to strong patterns in the PaySim dataset - fraud transactions are significantly larger than normal ones.

## Dataset

Using the PaySim synthetic financial dataset:
- 100,000 transactions
- ~8% fraud rate
- 7 features after preprocessing

## Project Structure

```
xai-fincrime-poc-starter/
├── backend/
│   ├── app/
│   │   ├── models/          # Model implementations
│   │   ├── explainers/      # SHAP explainer
│   │   └── main.py          # FastAPI app
│   └── train_*.py           # Training scripts
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── data/
│   ├── models/              # Trained model files
│   └── explanations/        # SHAP visualizations
└── archive/                 # Old 5-model ensemble
```

## How to Run

### Requirements
- Python 3.10+
- See requirements.txt for dependencies

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (if not already trained)
cd backend
python train_models.py

# Run the API
uvicorn app.main:app --reload

# Run the dashboard (separate terminal)
cd dashboard
streamlit run app.py
```

### Quick Test

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 50000, "type": "TRANSFER", "oldbalanceOrg": 50000}'
```

## Tech Stack

- Python, FastAPI (backend)
- Scikit-learn, XGBoost, PyTorch Geometric (ML)
- SHAP (explainability)
- Streamlit (dashboard)
- Pandas, NumPy (data processing)

## Known Issues / Limitations

1. Dataset is synthetic - real fraud patterns might be different
2. GNN attention visualization not fully implemented yet
3. Model doesn't update in real-time (static training)

## What I Learned

- Data leakage is a serious issue - I had to fix 3 leakage sources that were giving unrealistic 100% accuracy
- LSTM didn't work well for this dataset (dropped to 50% after fixing leakage) - turns out the dataset doesn't have strong sequential patterns
- Sometimes simpler is better - the 3-model ensemble is easier to explain than the original 5-model version

## References

1. Lundberg & Lee (2017) - SHAP
2. Lopez-Rojas et al. (2016) - PaySim dataset
3. Velickovic et al. (2018) - Graph Attention Networks

## Author

[Oshadha Ranatunga]  
[APIIT (Staffordshire University)]  
Final Year Project 2024/2025
---

*Note: Model files are not included in the repo due to size. Run the training scripts to generate them.*
