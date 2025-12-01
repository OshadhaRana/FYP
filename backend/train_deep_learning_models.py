"""
Comprehensive training script for all fraud detection models (RF, XGBoost, LSTM, CNN)
This script trains all models and saves them for the MVP deployment.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.preprocessing import TransactionPreprocessor
from app.utils.sequence_preprocessing import SequencePreprocessor, TemporalFeatureEngineer
from app.models.ml_models import FraudDetectionEnsemble
from app.models.deep_learning_models import LSTMFraudDetector, CNN1DFraudDetector, MultiModelEnsemble


def load_and_prepare_data(data_path: str):
    """Load and prepare the dataset for training"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['isFraud'])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['isFraud'])

    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Validation set: {val_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")

    return train_df, val_df, test_df


def train_traditional_models(train_df, val_df, test_df, models_dir):
    """Train Random Forest and XGBoost models"""
    print("\n" + "="*80)
    print("TRAINING TRADITIONAL MODELS (Random Forest + XGBoost)")
    print("="*80)

    # Initialize preprocessor
    preprocessor = TransactionPreprocessor()

    # Fit preprocessor on training data
    X_train, y_train = preprocessor.transform(train_df, fit=True)
    X_val, y_val = preprocessor.transform(val_df, fit=False)
    X_test, y_test = preprocessor.transform(test_df, fit=False)

    print(f"Training features shape: {X_train.shape}")

    # Initialize and train ensemble
    ensemble = FraudDetectionEnsemble(preprocessor)

    print("\nTraining Random Forest...")
    ensemble.rf_model.fit(X_train, y_train)

    print("Training XGBoost...")
    ensemble.xgb_model.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_pred = ensemble.predict(val_df)
    y_val_proba = ensemble.predict_proba(val_df).flatten()

    print("\n--- Validation Set Performance ---")
    print(classification_report(y_val, y_val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

    # Save models
    rf_path = os.path.join(models_dir, 'rf_model.joblib')
    xgb_path = os.path.join(models_dir, 'xgb_model.joblib')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')

    joblib.dump(ensemble.rf_model, rf_path)
    joblib.dump(ensemble.xgb_model, xgb_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"\nSaved Random Forest to: {rf_path}")
    print(f"Saved XGBoost to: {xgb_path}")
    print(f"Saved Preprocessor to: {preprocessor_path}")

    return ensemble, preprocessor, X_test, y_test


def train_lstm_model(train_df, val_df, test_df, models_dir):
    """Train LSTM model for sequential fraud detection"""
    print("\n" + "="*80)
    print("TRAINING LSTM MODEL (Sequential Pattern Detection)")
    print("="*80)

    # Initialize sequence preprocessor
    sequence_length = 10
    feature_cols = [
        'amount', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'hour', 'day', 'type_encoded'
    ]

    seq_preprocessor = SequencePreprocessor(sequence_length=sequence_length)
    temporal_engineer = TemporalFeatureEngineer()

    # Engineer temporal features
    print("Engineering temporal features...")
    train_df_temporal = temporal_engineer.fit_transform(train_df.copy())
    val_df_temporal = temporal_engineer.transform(val_df.copy())
    test_df_temporal = temporal_engineer.transform(test_df.copy())

    # Update feature columns with temporal features
    temporal_features = ['hour_of_day', 'day_of_month', 'is_weekend', 'is_night',
                         'time_since_last_txn', 'rolling_avg_amount', 'rolling_std_amount',
                         'balance_change_rate', 'amount_deviation']
    extended_features = feature_cols + temporal_features

    # Create sequences
    print("Creating sequences...")
    X_train_seq, y_train_seq = seq_preprocessor.fit_transform(train_df_temporal, extended_features)
    X_val_seq, y_val_seq = seq_preprocessor.transform(val_df_temporal, extended_features)
    X_test_seq, y_test_seq = seq_preprocessor.transform(test_df_temporal, extended_features)

    print(f"Training sequences shape: {X_train_seq.shape}")
    print(f"Validation sequences shape: {X_val_seq.shape}")

    # Initialize LSTM model
    n_features = X_train_seq.shape[2]
    lstm_model = LSTMFraudDetector(sequence_length=sequence_length, n_features=n_features)
    lstm_model.build_model()

    # Train LSTM
    print("\nTraining LSTM...")
    history = lstm_model.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=50,
        batch_size=128
    )

    # Evaluate on validation set
    y_val_pred_proba = lstm_model.predict(X_val_seq)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()

    print("\n--- LSTM Validation Set Performance ---")
    print(classification_report(y_val_seq, y_val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val_seq, y_val_pred_proba):.4f}")

    # Save LSTM model
    lstm_path = os.path.join(models_dir, 'lstm_model')
    seq_preprocessor_path = os.path.join(models_dir, 'sequence_preprocessor.joblib')
    temporal_engineer_path = os.path.join(models_dir, 'temporal_engineer.joblib')

    lstm_model.save(lstm_path)
    joblib.dump(seq_preprocessor, seq_preprocessor_path)
    joblib.dump(temporal_engineer, temporal_engineer_path)

    print(f"\nSaved LSTM model to: {lstm_path}")
    print(f"Saved Sequence Preprocessor to: {seq_preprocessor_path}")
    print(f"Saved Temporal Engineer to: {temporal_engineer_path}")

    return lstm_model, seq_preprocessor, temporal_engineer, X_test_seq, y_test_seq


def train_cnn_model(train_df, val_df, test_df, seq_preprocessor, temporal_engineer, models_dir):
    """Train 1D CNN model for feature extraction"""
    print("\n" + "="*80)
    print("TRAINING CNN MODEL (1D Feature Extraction)")
    print("="*80)

    sequence_length = 10
    feature_cols = [
        'amount', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'hour', 'day', 'type_encoded'
    ]

    # Use same temporal features as LSTM
    temporal_features = ['hour_of_day', 'day_of_month', 'is_weekend', 'is_night',
                         'time_since_last_txn', 'rolling_avg_amount', 'rolling_std_amount',
                         'balance_change_rate', 'amount_deviation']
    extended_features = feature_cols + temporal_features

    # Engineer temporal features
    print("Engineering temporal features...")
    train_df_temporal = temporal_engineer.transform(train_df.copy())
    val_df_temporal = temporal_engineer.transform(val_df.copy())
    test_df_temporal = temporal_engineer.transform(test_df.copy())

    # Create sequences (reuse seq_preprocessor)
    print("Creating sequences...")
    X_train_seq, y_train_seq = seq_preprocessor.transform(train_df_temporal, extended_features)
    X_val_seq, y_val_seq = seq_preprocessor.transform(val_df_temporal, extended_features)
    X_test_seq, y_test_seq = seq_preprocessor.transform(test_df_temporal, extended_features)

    print(f"Training sequences shape: {X_train_seq.shape}")

    # Initialize CNN model
    n_features = X_train_seq.shape[2]
    cnn_model = CNN1DFraudDetector(sequence_length=sequence_length, n_features=n_features)
    cnn_model.build_model()

    # Train CNN
    print("\nTraining CNN...")
    history = cnn_model.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=50,
        batch_size=128
    )

    # Evaluate on validation set
    y_val_pred_proba = cnn_model.predict(X_val_seq)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()

    print("\n--- CNN Validation Set Performance ---")
    print(classification_report(y_val_seq, y_val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val_seq, y_val_pred_proba):.4f}")

    # Save CNN model
    cnn_path = os.path.join(models_dir, 'cnn_model')
    cnn_model.save(cnn_path)

    print(f"\nSaved CNN model to: {cnn_path}")

    return cnn_model, X_test_seq, y_test_seq


def evaluate_ensemble(rf_model, xgb_model, lstm_model, cnn_model,
                      preprocessor, seq_preprocessor, temporal_engineer,
                      test_df, X_test_seq, y_test_seq, models_dir):
    """Evaluate the multi-model ensemble"""
    print("\n" + "="*80)
    print("EVALUATING MULTI-MODEL ENSEMBLE")
    print("="*80)

    # Create ensemble
    ensemble = MultiModelEnsemble(rf_model, xgb_model, lstm_model, cnn_model)

    # Prepare test data
    X_test_tabular, y_test = preprocessor.transform(test_df, fit=False)

    # Get ensemble predictions
    print("Generating ensemble predictions...")
    predictions_dict = ensemble.predict_proba(X_test_tabular, X_test_seq)

    ensemble_proba = predictions_dict['ensemble'].flatten()
    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    print("\n--- Ensemble Test Set Performance ---")
    print(classification_report(y_test, ensemble_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, ensemble_proba):.4f}")

    print("\n--- Individual Model Contributions ---")
    for model_name in ['rf', 'xgb', 'lstm', 'cnn']:
        model_proba = predictions_dict[model_name].flatten()
        model_auc = roc_auc_score(y_test, model_proba)
        print(f"{model_name.upper()}: ROC-AUC = {model_auc:.4f}, Weight = {ensemble.weights[model_name]:.2f}")

    # Save ensemble weights configuration
    config_path = os.path.join(models_dir, 'ensemble_config.json')
    config = {
        'weights': ensemble.weights,
        'sequence_length': seq_preprocessor.sequence_length,
        'feature_columns': [
            'amount', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'hour', 'day', 'type_encoded'
        ],
        'temporal_features': ['hour_of_day', 'day_of_month', 'is_weekend', 'is_night',
                              'time_since_last_txn', 'rolling_avg_amount', 'rolling_std_amount',
                              'balance_change_rate', 'amount_deviation']
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved ensemble configuration to: {config_path}")

    return ensemble


def main():
    """Main training pipeline"""
    print("="*80)
    print("FRAUD DETECTION MVP - DEEP LEARNING MODEL TRAINING")
    print("="*80)

    # Paths
    data_path = r"C:\xai-fincrime-poc-starter\xai-fincrime-poc\data\processed\paysim_sample.csv"
    models_dir = r"C:\xai-fincrime-poc-starter\xai-fincrime-poc\data\models"

    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_and_prepare_data(data_path)

    # Train traditional models (RF + XGBoost)
    traditional_ensemble, preprocessor, X_test_tabular, y_test = train_traditional_models(
        train_df, val_df, test_df, models_dir
    )

    # Train LSTM model
    lstm_model, seq_preprocessor, temporal_engineer, X_test_seq_lstm, y_test_seq_lstm = train_lstm_model(
        train_df, val_df, test_df, models_dir
    )

    # Train CNN model (reuses seq_preprocessor and temporal_engineer)
    cnn_model, X_test_seq_cnn, y_test_seq_cnn = train_cnn_model(
        train_df, val_df, test_df, seq_preprocessor, temporal_engineer, models_dir
    )

    # Evaluate multi-model ensemble
    ensemble = evaluate_ensemble(
        traditional_ensemble.rf_model,
        traditional_ensemble.xgb_model,
        lstm_model,
        cnn_model,
        preprocessor,
        seq_preprocessor,
        temporal_engineer,
        test_df,
        X_test_seq_lstm,
        y_test_seq_lstm,
        models_dir
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll models saved to: {models_dir}")
    print("\nModel files:")
    print("  - rf_model.joblib")
    print("  - xgb_model.joblib")
    print("  - lstm_model/")
    print("  - cnn_model/")
    print("  - preprocessor.joblib")
    print("  - sequence_preprocessor.joblib")
    print("  - temporal_engineer.joblib")
    print("  - ensemble_config.json")


if __name__ == "__main__":
    main()
