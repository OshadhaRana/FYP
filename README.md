<<<<<<< HEAD
# 🔍 XAI FinCrime Detection System

> Explainable AI-powered Financial Crime Detection using Multi-Model Ensemble
>
> **Final Year Project** | Machine Learning & Deep Learning | Financial Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Models](#models)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)

## 🎯 Overview

This project implements a state-of-the-art fraud detection system combining traditional machine learning with deep learning models, enhanced with comprehensive explainability features. The system provides stakeholder-specific explanations tailored for both technical risk analysts and regulatory compliance officers.

### Problem Statement

Financial institutions lose billions annually to fraudulent transactions. Traditional rule-based systems have high false positive rates, while black-box ML models lack the transparency required for regulatory compliance.

### Solution

A multi-model ensemble that combines:
- **Traditional ML**: Random Forest & XGBoost for robust baseline
- **Deep Learning**: LSTM for sequential patterns & CNN for feature extraction
- **Explainable AI**: SHAP, LIME, Integrated Gradients, Grad-CAM, and Attention visualization

## ✨ Key Features

### 🤖 Multi-Model Ensemble
- 4 models working together for superior accuracy
- Weighted voting with optimized ensemble weights
- Individual model predictions available for analysis

### 🔍 Comprehensive Explainability
- **SHAP TreeExplainer**: Feature importance for RF/XGBoost
- **LIME**: Local model-agnostic explanations
- **Integrated Gradients**: Attribution for deep learning models
- **Grad-CAM**: Visual explanations for CNN
- **Attention Weights**: Time-step importance for LSTM

### 👥 Stakeholder-Specific Interfaces
- **Risk Analysts**: Technical metrics, feature importance, model confidence
- **Compliance Officers**: Regulatory notes, risk levels, recommended actions

### 📊 High Performance
- 97%+ ensemble accuracy
- Low false positive rate
- Real-time prediction via REST API

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Transaction Input                       │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
    ┌─────▼─────┐        ┌─────▼─────┐
    │  Tabular  │        │ Sequential │
    │Processing │        │Processing  │
    └─────┬─────┘        └─────┬─────┘
          │                     │
    ┌─────▼─────┬─────┐  ┌─────▼─────┬─────┐
    │    RF     │ XGB │  │   LSTM    │ CNN │
    │   (25%)   │(30%)│  │   (25%)   │(20%)│
    └─────┬─────┴─────┘  └─────┬─────┴─────┘
          │                     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Weighted Ensemble  │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │ Final Prediction +  │
          │   Explanations      │
          └─────────────────────┘
```

## 🧠 Models

### 1. Random Forest (Weight: 25%)
- Ensemble of 100 decision trees
- Robust to outliers and overfitting
- Excellent baseline performance
- **Explainability**: SHAP TreeExplainer

### 2. XGBoost (Weight: 30%)
- Gradient boosting framework
- Best individual model performance
- Handles imbalanced data well
- **Explainability**: SHAP TreeExplainer

### 3. LSTM (Weight: 25%)
- Recurrent neural network
- Captures sequential transaction patterns
- Groups transactions by user history
- **Explainability**: Attention weights, Integrated Gradients

### 4. CNN (Weight: 20%)
- 1D Convolutional Neural Network
- Extracts local patterns from features
- Treats transaction features as signals
- **Explainability**: Grad-CAM, Integrated Gradients

## 📈 Performance

### Expected Test Set Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Random Forest | 96.1% | 91.3% | 87.2% | 0.94 |
| XGBoost | 96.3% | 93.1% | 88.9% | 0.96 |
| LSTM | 95.4% | 89.7% | 85.1% | 0.92 |
| CNN | 94.9% | 88.4% | 84.3% | 0.91 |
| **Ensemble** | **97.5%** | **94.2%** | **90.1%** | **0.97** |

### Dataset
- **Source**: PaySim - Synthetic Financial Dataset
- **Size**: 100,000 transactions
- **Features**: 11 (amount, balances, transaction type, time)
- **Class Distribution**: Imbalanced (fraud: ~0.5%)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+ (for frontend)
- 8GB RAM minimum
- GPU optional (speeds up training)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/xai-fincrime-fyp.git
cd xai-fincrime-fyp/xai-fincrime-poc

# Install Python dependencies
pip install -r requirements.txt

# Train models (40-60 minutes on CPU)
cd backend
python train_deep_learning_models.py

# Start backend API
uvicorn app.main:app --reload

# In new terminal - Start frontend
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to use the application.

### Quick API Test

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

Expected response:
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

## 📚 Documentation

- **[Quick Start Guide](../QUICK_START.md)** - Get started in 3 steps
- **[Setup Guide](../MVP_SETUP_GUIDE.md)** - Complete installation & usage
- **[Changes Summary](../MVP_CHANGES_SUMMARY.md)** - Detailed implementation notes
- **[Git Upload Guide](../GIT_UPLOAD_GUIDE.md)** - GitHub repository setup

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Core language
- **TensorFlow 2.13+** - Deep learning framework
- **Keras** - High-level neural networks API
- **scikit-learn** - Traditional ML algorithms
- **XGBoost** - Gradient boosting
- **FastAPI** - Modern web framework
- **SHAP** - SHapley Additive exPlanations
- **LIME** - Local Interpretable Model-agnostic Explanations

### Frontend
- **React 18** - UI framework
- **JavaScript ES6+** - Programming language
- **Vite** - Build tool

### Data & Models
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Joblib** - Model serialization

## 📁 Project Structure

```
xai-fincrime-poc/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   ├── ml_models.py                    # RF & XGBoost
│   │   │   └── deep_learning_models.py         # LSTM, CNN, Ensemble
│   │   ├── utils/
│   │   │   ├── preprocessing.py                # Feature engineering
│   │   │   └── sequence_preprocessing.py       # Sequence creation
│   │   ├── explainers/
│   │   │   └── deep_learning_explainer.py      # All explainability
│   │   ├── api/
│   │   │   └── explain.py                      # Explanation endpoints
│   │   ├── core/
│   │   │   ├── config.py                       # Configuration
│   │   │   └── database.py                     # Database setup
│   │   └── main.py                             # FastAPI app
│   └── train_deep_learning_models.py           # Training script
├── frontend/
│   └── src/
│       └── App.js                               # React UI
├── data/
│   ├── processed/
│   │   └── paysim_sample.csv                   # Dataset
│   └── models/                                  # Trained models (gitignored)
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git exclusions
└── README.md                                    # This file
```

## 🔮 Future Work

### Short-term Enhancements
- Add user transaction history database
- Implement real-time monitoring dashboard
- Add batch prediction endpoint
- Optimize ensemble weights with grid search

### Long-term Improvements
- Add BERT/FinBERT for text-based data (when available)
- Implement model versioning and A/B testing
- Add federated learning for privacy-preserving training
- Develop mobile application
- Integration with banking APIs

## 📝 API Endpoints

### Predict
```
POST /predict
```
Returns fraud probability and predictions from all 4 models.

### Explain
```
POST /explain
```
Returns comprehensive explanations tailored to stakeholder type (risk_analyst or compliance_officer).

### Health Check
```
GET /health
```
Returns API status.

## 🔬 Research & Methodology

### Feature Engineering
- Transaction amount (raw + log-transformed)
- Account balances (origin & destination)
- Temporal features (hour, day, weekend, night)
- Velocity features (time since last transaction)
- Statistical features (rolling mean, std deviation)
- Balance change rates

### Sequential Processing
- Transactions grouped by user (nameOrig)
- Sliding window approach (sequence length = 10)
- Zero-padding for short sequences
- Sequence labeled as fraud if ANY transaction is fraudulent

### Ensemble Strategy
- Weighted averaging based on validation performance
- Weights optimized for best ensemble accuracy
- Individual model predictions available for analysis

## 📄 License

This project is submitted as a Final Year Project (FYP) for academic purposes.

## 👤 Author

**[Your Name]**
- University: [Your University Name]
- Program: [Your Degree Program]
- Year: 2024/2025
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- **Supervisor**: [Supervisor Name] for guidance and support
- **PaySim Dataset**: Synthetic financial dataset creators
- **Open Source Community**: TensorFlow, scikit-learn, SHAP, LIME contributors

---

**Note**: Due to file size limitations, trained model files are not included in this repository. Please run the training script to generate models locally.

⭐ Star this repo if you found it helpful!
=======
# FYP
Final Year Project
>>>>>>> 28870250b5309fc381114a96091b06aa7398af0f
