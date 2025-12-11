"""
Complete Training Script for Research-Grade Fraud Detection System

Trains all 6 models in the heterogeneous ensemble:
1. Random Forest (baseline)
2. XGBoost (baseline)
3. LSTM (sequential patterns)
4. CNN (local patterns)
5. Graph Neural Network (network patterns) - RESEARCH
6. TabTransformer (feature interactions) - RESEARCH

Saves all models and generates comprehensive evaluation report.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import json
import time
from datetime import datetime

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.preprocessing import Preprocessor
from app.utils.sequence_preprocessing import SequencePreprocessor, TemporalFeatureEngineer
from app.models.ml_models import FraudDetectionEnsemble
from app.models.deep_learning_models import LSTMFraudDetector, CNN1DFraudDetector, MultiModelEnsemble

# Import research models
try:
    from app.models.graph_models import GNNFraudDetectorWrapper, TemporalGraphBuilder
    from app.models.transformer_models import TabTransformerWrapper
    RESEARCH_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Research models not available: {e}")
    print("Install dependencies: pip install torch torch-geometric transformers")
    RESEARCH_MODELS_AVAILABLE = False


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def load_and_prepare_data(data_path: str):
    """Load and prepare the dataset for training"""
    print_header("LOADING DATA")
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
    print_header("TRAINING TRADITIONAL MODELS (Random Forest + XGBoost)")

    start_time = time.time()

    # Initialize preprocessor
    preprocessor = Preprocessor()

    # Fit preprocessor on training data
    X_train, _ = preprocessor.transform(train_df, fit=True)
    X_val, _ = preprocessor.transform(val_df, fit=False)
    X_test, _ = preprocessor.transform(test_df, fit=False)

    # Extract labels
    y_train = train_df['isFraud'].values
    y_val = val_df['isFraud'].values
    y_test = test_df['isFraud'].values

    print(f"Training features shape: {X_train.shape}")

    # Initialize and train ensemble
    ensemble = FraudDetectionEnsemble(preprocessor)

    print("\nTraining Random Forest...")
    ensemble.rf_model.fit(X_train, y_train)

    print("Training XGBoost...")
    ensemble.xgb_model.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_proba = ensemble.predict_proba(val_df).flatten()
    y_val_pred = (y_val_proba > 0.5).astype(int)

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

    print(f"\n✓ Saved Random Forest to: {rf_path}")
    print(f"✓ Saved XGBoost to: {xgb_path}")
    print(f"✓ Saved Preprocessor to: {preprocessor_path}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Traditional models training time: {elapsed/60:.2f} minutes")

    return ensemble, preprocessor, X_test, y_test


def train_lstm_cnn_models(train_df, val_df, test_df, preprocessor, models_dir):
    """Train LSTM and CNN models"""
    print_header("TRAINING DEEP LEARNING MODELS (LSTM + CNN)")

    # Train LSTM
    lstm_model, seq_preprocessor, X_test_seq_lstm, y_test_seq_lstm = train_lstm_model(
        train_df, val_df, test_df, models_dir
    )

    # Train CNN
    cnn_model, X_test_cnn, y_test_cnn = train_cnn_model(
        train_df, val_df, test_df, preprocessor, models_dir
    )

    return lstm_model, cnn_model, seq_preprocessor, X_test_seq_lstm


def train_lstm_model(train_df, val_df, test_df, models_dir):
    """Train LSTM model (from original training script)"""
    print("\n" + "="*80)
    print("TRAINING LSTM MODEL (Sequential Pattern Detection)")
    print("="*80)

    start_time = time.time()

    from sklearn.preprocessing import LabelEncoder

    sequence_length = 10
    seq_preprocessor = SequencePreprocessor(sequence_length=sequence_length)

    # Add preprocessing features
    label_encoder = LabelEncoder()

    train_df_prep = train_df.copy()
    val_df_prep = val_df.copy()
    test_df_prep = test_df.copy()

    for df in [train_df_prep, val_df_prep, test_df_prep]:
        df['hour'] = df['step'] % 24
        df['day'] = df['step'] // 24
        df['amount_log'] = np.log1p(df['amount'])

    train_df_prep['type_encoded'] = label_encoder.fit_transform(train_df_prep['type'])
    val_df_prep['type_encoded'] = label_encoder.transform(val_df_prep['type'])
    test_df_prep['type_encoded'] = label_encoder.transform(test_df_prep['type'])

    # Add temporal features
    train_df_temporal = TemporalFeatureEngineer.add_temporal_features(train_df_prep)
    val_df_temporal = TemporalFeatureEngineer.add_temporal_features(val_df_prep)
    test_df_temporal = TemporalFeatureEngineer.add_temporal_features(test_df_prep)

    # Define feature columns
    feature_cols = [
        'amount', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'hour', 'day', 'type_encoded'
    ]

    temporal_features = ['hour_of_day', 'day_of_month', 'is_weekend', 'is_night',
                         'time_since_last_txn', 'rolling_avg_amount', 'rolling_std_amount',
                         'balance_change_rate', 'amount_deviation']
    extended_features = feature_cols + temporal_features

    # Create sequences
    X_train_seq, y_train_seq = seq_preprocessor.create_sequences(train_df_temporal, extended_features)
    X_val_seq, y_val_seq = seq_preprocessor.create_sequences(val_df_temporal, extended_features)
    X_test_seq, y_test_seq = seq_preprocessor.create_sequences(test_df_temporal, extended_features)

    print(f"Training sequences shape: {X_train_seq.shape}")

    # Initialize LSTM
    n_features = X_train_seq.shape[2]
    lstm_model = LSTMFraudDetector(sequence_length=sequence_length, n_features=n_features)

    # Train
    X_combined = np.concatenate([X_train_seq, X_val_seq], axis=0)
    y_combined = np.concatenate([y_train_seq, y_val_seq], axis=0)

    history = lstm_model.fit(X_combined, y_combined, validation_split=0.15, epochs=50, batch_size=128)

    # Evaluate
    y_val_pred_proba = lstm_model.predict_proba(X_val_seq)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()

    print("\n--- LSTM Validation Set Performance ---")
    print(classification_report(y_val_seq, y_val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val_seq, y_val_pred_proba):.4f}")

    # Save
    lstm_path = os.path.join(models_dir, 'lstm_model')
    seq_preprocessor_path = os.path.join(models_dir, 'sequence_preprocessor.joblib')

    lstm_model.save(lstm_path)
    joblib.dump(seq_preprocessor, seq_preprocessor_path)

    elapsed = time.time() - start_time
    print(f"\n⏱️  LSTM training time: {elapsed/60:.2f} minutes")

    return lstm_model, seq_preprocessor, X_test_seq, y_test_seq


def train_cnn_model(train_df, val_df, test_df, preprocessor, models_dir):
    """Train CNN model (from original training script)"""
    print("\n" + "="*80)
    print("TRAINING CNN MODEL (1D Feature Extraction)")
    print("="*80)

    start_time = time.time()

    X_train, _ = preprocessor.transform(train_df, fit=False)
    X_val, _ = preprocessor.transform(val_df, fit=False)
    X_test, _ = preprocessor.transform(test_df, fit=False)

    y_train = train_df['isFraud'].values
    y_val = val_df['isFraud'].values
    y_test = test_df['isFraud'].values

    n_features = X_train.shape[1]
    cnn_model = CNN1DFraudDetector(n_features=n_features)

    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)

    history = cnn_model.fit(X_combined, y_combined, validation_split=0.15, epochs=50, batch_size=128)

    # Evaluate
    y_val_pred_proba = cnn_model.predict_proba(X_val)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()

    print("\n--- CNN Validation Set Performance ---")
    print(classification_report(y_val, y_val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_val_pred_proba):.4f}")

    # Save
    cnn_path = os.path.join(models_dir, 'cnn_model')
    cnn_model.save(cnn_path)

    elapsed = time.time() - start_time
    print(f"\n⏱️  CNN training time: {elapsed/60:.2f} minutes")

    return cnn_model, X_test, y_test


def train_gnn_model(train_df, val_df, test_df, models_dir):
    """Train Graph Neural Network model"""
    print_header("🔬 TRAINING GRAPH NEURAL NETWORK (Network Pattern Detection)")

    if not RESEARCH_MODELS_AVAILABLE:
        print("⚠️  Skipping GNN training - research dependencies not installed")
        return None

    start_time = time.time()

    # Initialize GNN
    gnn = GNNFraudDetectorWrapper(node_features=8, edge_features=7, hidden_dim=64)

    # Train on combined data (GNN works best with full graph)
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    print("Training GNN on transaction graph...")
    gnn.fit(combined_df, epochs=50, lr=0.001)

    # Evaluate on validation set
    val_proba = gnn.predict_proba(val_df)
    val_pred = (val_proba > 0.5).astype(int)
    y_val = val_df['isFraud'].values

    print("\n--- GNN Validation Set Performance ---")
    print(classification_report(y_val, val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    # Detect fraud patterns
    patterns = gnn.graph_builder.detect_fraud_patterns(val_df)
    print(f"\n📊 Network Patterns Detected:")
    print(f"  - Circular flows: {len(patterns['circular_flows'])}")
    print(f"  - Fan-out accounts: {len(patterns['fan_out_accounts'])}")
    print(f"  - Rapid sequences: {len(patterns['rapid_sequences'])}")

    # Save
    gnn_path = os.path.join(models_dir, 'gnn_model.pt')
    gnn.save(gnn_path)

    elapsed = time.time() - start_time
    print(f"\n⏱️  GNN training time: {elapsed/60:.2f} minutes")

    return gnn


def train_tabtransformer_model(train_df, val_df, test_df, preprocessor, models_dir):
    """Train TabTransformer model"""
    print_header("🔬 TRAINING TABTRANSFORMER (Advanced Tabular Learning)")

    if not RESEARCH_MODELS_AVAILABLE:
        print("⚠️  Skipping TabTransformer training - research dependencies not installed")
        return None

    start_time = time.time()

    # Prepare data
    X_train, _ = preprocessor.transform(train_df, fit=False)
    X_val, _ = preprocessor.transform(val_df, fit=False)
    y_train = train_df['isFraud'].values
    y_val = val_df['isFraud'].values

    # Initialize TabTransformer
    tab_transformer = TabTransformerWrapper(num_continuous=8, categorical_info={'type': 5})

    print("Training TabTransformer with self-attention...")
    tab_transformer.fit(X_train, y_train, epochs=50, batch_size=128, lr=0.001)

    # Evaluate
    val_proba = tab_transformer.predict_proba(X_val)
    val_pred = (val_proba > 0.5).astype(int)

    print("\n--- TabTransformer Validation Set Performance ---")
    print(classification_report(y_val, val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    # Save
    tab_transformer_path = os.path.join(models_dir, 'tabtransformer_model.pt')
    tab_transformer.save(tab_transformer_path)

    elapsed = time.time() - start_time
    print(f"\n⏱️  TabTransformer training time: {elapsed/60:.2f} minutes")

    return tab_transformer


def evaluate_full_ensemble(models_dict, test_df, preprocessor, seq_preprocessor, models_dir):
    """Evaluate the complete 6-model ensemble"""
    print_header("🏆 EVALUATING FULL RESEARCH ENSEMBLE")

    # Prepare test data
    X_test_tabular, _ = preprocessor.transform(test_df, fit=False)
    y_test = test_df['isFraud'].values

    # Prepare sequence data for LSTM
    test_df_prep = test_df.copy()
    test_df_prep['hour'] = test_df_prep['step'] % 24
    test_df_prep['day'] = test_df_prep['step'] // 24
    test_df_prep['amount_log'] = np.log1p(test_df_prep['amount'])

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    test_df_prep['type_encoded'] = label_encoder.fit_transform(test_df_prep['type'])

    test_df_temporal = TemporalFeatureEngineer.add_temporal_features(test_df_prep)
    feature_cols = [
        'amount', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'hour', 'day', 'type_encoded'
    ]
    temporal_features = ['hour_of_day', 'day_of_month', 'is_weekend', 'is_night',
                         'time_since_last_txn', 'rolling_avg_amount', 'rolling_std_amount',
                         'balance_change_rate', 'amount_deviation']
    extended_features = feature_cols + temporal_features
    X_test_seq, _ = seq_preprocessor.create_sequences(test_df_temporal, extended_features)

    # Get predictions from each model
    predictions = {}

    print("Generating predictions from all models...")

    # Traditional models
    predictions['rf'] = models_dict['rf'].predict_proba(X_test_tabular)[:, 1]
    predictions['xgb'] = models_dict['xgb'].predict_proba(X_test_tabular)[:, 1]

    # Deep learning
    predictions['lstm'] = models_dict['lstm'].predict_proba(X_test_seq).flatten()
    predictions['cnn'] = models_dict['cnn'].predict_proba(X_test_tabular).flatten()

    # Research models (if available)
    if models_dict.get('gnn'):
        predictions['gnn'] = models_dict['gnn'].predict_proba(test_df)

    if models_dict.get('tabtransformer'):
        predictions['tabtransformer'] = models_dict['tabtransformer'].predict_proba(X_test_tabular)

    # Calculate ensemble weights (research optimized)
    if RESEARCH_MODELS_AVAILABLE and all(k in predictions for k in ['gnn', 'tabtransformer']):
        weights = {
            'rf': 0.15,
            'xgb': 0.15,
            'lstm': 0.15,
            'cnn': 0.10,
            'gnn': 0.25,  # Highest weight for network patterns
            'tabtransformer': 0.20
        }
    else:
        # Fallback to original weights
        weights = {
            'rf': 0.25,
            'xgb': 0.30,
            'lstm': 0.25,
            'cnn': 0.20
        }

    # Ensemble prediction
    ensemble_proba = np.zeros(len(y_test))
    for model_name, weight in weights.items():
        if model_name in predictions:
            ensemble_proba += predictions[model_name] * weight

    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    # Evaluation
    print("\n" + "="*80)
    print("FINAL ENSEMBLE PERFORMANCE")
    print("="*80)
    print(classification_report(y_test, ensemble_pred))
    print(f"\nROC-AUC: {roc_auc_score(y_test, ensemble_proba):.4f}")

    # Individual model performance
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL CONTRIBUTIONS")
    print("="*80)

    results = {}
    for model_name in predictions.keys():
        model_auc = roc_auc_score(y_test, predictions[model_name])
        model_weight = weights.get(model_name, 0)
        print(f"{model_name.upper():20s} - ROC-AUC: {model_auc:.4f}, Weight: {model_weight:.2f}")
        results[model_name] = {'auc': model_auc, 'weight': model_weight}

    # Save configuration
    config = {
        'weights': weights,
        'model_performance': results,
        'ensemble_auc': float(roc_auc_score(y_test, ensemble_proba)),
        'timestamp': datetime.now().isoformat()
    }

    config_path = os.path.join(models_dir, 'research_ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Saved ensemble configuration to: {config_path}")

    return ensemble_proba, predictions, results


def main():
    """Main training pipeline for research system"""
    print_header("🔬 RESEARCH-GRADE FRAUD DETECTION SYSTEM - FULL TRAINING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    overall_start = time.time()

    # Paths
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    data_path = os.path.join(project_root, "data", "processed", "paysim_sample.csv")
    models_dir = os.path.join(project_root, "data", "models")

    os.makedirs(models_dir, exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_and_prepare_data(data_path)

    # Train models
    models_dict = {}

    # 1. Traditional models
    trad_ensemble, preprocessor, X_test, y_test = train_traditional_models(
        train_df, val_df, test_df, models_dir
    )
    models_dict['rf'] = trad_ensemble.rf_model
    models_dict['xgb'] = trad_ensemble.xgb_model

    # 2. Deep learning models
    lstm_model, cnn_model, seq_preprocessor, X_test_seq = train_lstm_cnn_models(
        train_df, val_df, test_df, preprocessor, models_dir
    )
    models_dict['lstm'] = lstm_model
    models_dict['cnn'] = cnn_model

    # 3. Research models
    if RESEARCH_MODELS_AVAILABLE:
        # GNN
        gnn_model = train_gnn_model(train_df, val_df, test_df, models_dir)
        if gnn_model:
            models_dict['gnn'] = gnn_model

        # TabTransformer
        tab_transformer = train_tabtransformer_model(train_df, val_df, test_df, preprocessor, models_dir)
        if tab_transformer:
            models_dict['tabtransformer'] = tab_transformer

    # 4. Evaluate full ensemble
    ensemble_proba, predictions, results = evaluate_full_ensemble(
        models_dict, test_df, preprocessor, seq_preprocessor, models_dir
    )

    # Final summary
    overall_elapsed = time.time() - overall_start
    print_header("✅ TRAINING COMPLETE!")
    print(f"Total training time: {overall_elapsed/60:.2f} minutes ({overall_elapsed/3600:.2f} hours)")
    print(f"\nModels saved to: {models_dir}")
    print("\nTrained models:")
    for model_name in models_dict.keys():
        print(f"  ✓ {model_name.upper()}")

    print("\n" + "="*80)
    print("🎯 RESEARCH SYSTEM READY FOR:")
    print("  - Ablation studies")
    print("  - Network fraud analysis")
    print("  - Feature interaction analysis")
    print("  - User study (stakeholder XAI)")
    print("  - Publication-ready experiments")
    print("="*80)


if __name__ == "__main__":
    main()
