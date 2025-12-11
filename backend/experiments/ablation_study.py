"""
Ablation Study: Systematic evaluation of each model's contribution

Research Question: How much does each model contribute to overall performance?

Methodology:
1. Test each model individually (baselines)
2. Test cumulative additions (baseline → +LSTM → +CNN → +GNN → +TabTransformer)
3. Test model removal (full ensemble minus one model)
4. Statistical significance testing

Output: Performance table showing contribution of each component
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.preprocessing import Preprocessor
from app.utils.sequence_preprocessing import SequencePreprocessor, TemporalFeatureEngineer


def load_test_data():
    """Load preprocessed test data"""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(backend_dir)
    data_path = os.path.join(project_root, "data", "processed", "paysim_sample.csv")

    df = pd.read_csv(data_path)

    # Use same split as training (20% test)
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['isFraud'])

    return test_df


def load_models(models_dir):
    """Load all trained models"""
    models = {}

    print("Loading models...")

    # Traditional models
    models['rf'] = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
    models['xgb'] = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))

    # Preprocessors
    models['preprocessor'] = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
    models['seq_preprocessor'] = joblib.load(os.path.join(models_dir, 'sequence_preprocessor.joblib'))

    # Deep learning models
    from app.models.deep_learning_models import LSTMFraudDetector, CNN1DFraudDetector
    models['lstm'] = LSTMFraudDetector.load(os.path.join(models_dir, 'lstm_model'))
    models['cnn'] = CNN1DFraudDetector.load(os.path.join(models_dir, 'cnn_model'))

    # Research models (if available)
    try:
        from app.models.graph_models import GNNFraudDetectorWrapper
        from app.models.transformer_models import TabTransformerWrapper

        gnn_path = os.path.join(models_dir, 'gnn_model.pt')
        if os.path.exists(gnn_path):
            models['gnn'] = GNNFraudDetectorWrapper.load(gnn_path)
            print("  ✓ GNN loaded")

        tab_path = os.path.join(models_dir, 'tabtransformer_model.pt')
        if os.path.exists(tab_path):
            models['tabtransformer'] = TabTransformerWrapper.load(tab_path)
            print("  ✓ TabTransformer loaded")
    except Exception as e:
        print(f"  ⚠️  Research models not available: {e}")

    return models


def prepare_test_features(test_df, preprocessor, seq_preprocessor):
    """Prepare all test features"""
    # Tabular features
    X_test_tabular, _ = preprocessor.transform(test_df, fit=False)

    # Sequential features
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

    y_test = test_df['isFraud'].values

    return X_test_tabular, X_test_seq, test_df, y_test


def get_predictions(models, X_test_tabular, X_test_seq, test_df):
    """Get predictions from all models"""
    predictions = {}

    # Traditional models
    predictions['rf'] = models['rf'].predict_proba(X_test_tabular)[:, 1]
    predictions['xgb'] = models['xgb'].predict_proba(X_test_tabular)[:, 1]

    # Deep learning
    predictions['lstm'] = models['lstm'].predict_proba(X_test_seq).flatten()
    predictions['cnn'] = models['cnn'].predict_proba(X_test_tabular).flatten()

    # Research models
    if 'gnn' in models:
        predictions['gnn'] = models['gnn'].predict_proba(test_df)

    if 'tabtransformer' in models:
        predictions['tabtransformer'] = models['tabtransformer'].predict_proba(X_test_tabular)

    return predictions


def evaluate_configuration(predictions, weights, y_test):
    """Evaluate a specific model configuration"""
    ensemble_proba = np.zeros(len(y_test))

    for model_name, weight in weights.items():
        if model_name in predictions:
            ensemble_proba += predictions[model_name] * weight

    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'precision': precision_score(y_test, ensemble_pred),
        'recall': recall_score(y_test, ensemble_pred),
        'f1': f1_score(y_test, ensemble_pred),
        'roc_auc': roc_auc_score(y_test, ensemble_proba)
    }

    return metrics


def ablation_study_individual_models(predictions, y_test):
    """Test 1: Individual model performance"""
    print("\n" + "="*80)
    print("ABLATION STUDY 1: Individual Model Performance")
    print("="*80)

    results = []

    for model_name in predictions.keys():
        weights = {model_name: 1.0}
        metrics = evaluate_configuration(predictions, weights, y_test)

        results.append({
            'Configuration': model_name.upper(),
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Precision': f"{metrics['precision']*100:.2f}%",
            'Recall': f"{metrics['recall']*100:.2f}%",
            'F1': f"{metrics['f1']*100:.2f}%",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'num_models': 1
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    return df_results


def ablation_study_cumulative(predictions, y_test):
    """Test 2: Cumulative model addition"""
    print("\n" + "="*80)
    print("ABLATION STUDY 2: Cumulative Model Addition")
    print("="*80)

    results = []

    # Define cumulative configurations
    configs = [
        ({'rf': 1.0}, 'RF only'),
        ({'rf': 0.5, 'xgb': 0.5}, 'RF + XGB'),
        ({'rf': 0.33, 'xgb': 0.33, 'lstm': 0.34}, 'RF + XGB + LSTM'),
        ({'rf': 0.25, 'xgb': 0.25, 'lstm': 0.25, 'cnn': 0.25}, 'RF + XGB + LSTM + CNN'),
    ]

    # Add GNN if available
    if 'gnn' in predictions:
        configs.append(
            ({'rf': 0.2, 'xgb': 0.2, 'lstm': 0.2, 'cnn': 0.15, 'gnn': 0.25},
             'RF + XGB + LSTM + CNN + GNN')
        )

    # Add TabTransformer if available
    if 'tabtransformer' in predictions:
        configs.append(
            ({'rf': 0.15, 'xgb': 0.15, 'lstm': 0.15, 'cnn': 0.10,
              'gnn': 0.25, 'tabtransformer': 0.20},
             'Full System (All 6 Models)')
        )

    for weights, config_name in configs:
        metrics = evaluate_configuration(predictions, weights, y_test)

        results.append({
            'Configuration': config_name,
            'Models': sum(weights.values() > 0),
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Precision': f"{metrics['precision']*100:.2f}%",
            'Recall': f"{metrics['recall']*100:.2f}%",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'Δ AUC': ''
        })

    # Calculate improvements
    for i in range(1, len(results)):
        prev_auc = float(results[i-1]['ROC-AUC'])
        curr_auc = float(results[i]['ROC-AUC'])
        delta = curr_auc - prev_auc
        results[i]['Δ AUC'] = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    return df_results


def ablation_study_leave_one_out(predictions, y_test):
    """Test 3: Leave-one-out analysis"""
    print("\n" + "="*80)
    print("ABLATION STUDY 3: Leave-One-Out Analysis")
    print("="*80)

    # Full ensemble weights
    if 'gnn' in predictions and 'tabtransformer' in predictions:
        full_weights = {
            'rf': 0.15, 'xgb': 0.15, 'lstm': 0.15, 'cnn': 0.10,
            'gnn': 0.25, 'tabtransformer': 0.20
        }
    else:
        full_weights = {
            'rf': 0.25, 'xgb': 0.30, 'lstm': 0.25, 'cnn': 0.20
        }

    # Full ensemble performance
    full_metrics = evaluate_configuration(predictions, full_weights, y_test)
    full_auc = full_metrics['roc_auc']

    results = []

    # Leave each model out
    for model_to_remove in full_weights.keys():
        # Create weights without this model
        reduced_weights = {k: v for k, v in full_weights.items() if k != model_to_remove}

        # Renormalize
        total = sum(reduced_weights.values())
        reduced_weights = {k: v/total for k, v in reduced_weights.items()}

        metrics = evaluate_configuration(predictions, reduced_weights, y_test)

        impact = full_auc - metrics['roc_auc']

        results.append({
            'Removed Model': model_to_remove.upper(),
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'Impact (Δ AUC)': f"-{impact:.4f}" if impact > 0 else f"+{abs(impact):.4f}",
            'Importance': '⭐⭐⭐⭐⭐' if impact > 0.005 else '⭐⭐⭐' if impact > 0.002 else '⭐'
        })

    # Sort by impact
    results.sort(key=lambda x: float(x['Impact (Δ AUC)'].replace('-', '').replace('+', '')), reverse=True)

    df_results = pd.DataFrame(results)
    print(f"\nFull Ensemble ROC-AUC: {full_auc:.4f}")
    print(df_results.to_string(index=False))

    return df_results


def visualize_results(individual_df, cumulative_df, leave_one_out_df, output_dir):
    """Create visualization of ablation study results"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Individual model performance
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Parse ROC-AUC values
    models = individual_df['Configuration'].tolist()
    aucs = [float(x) for x in individual_df['ROC-AUC'].tolist()]

    axes[0].barh(models, aucs, color='steelblue')
    axes[0].set_xlabel('ROC-AUC Score')
    axes[0].set_title('Individual Model Performance')
    axes[0].set_xlim(0.9, 1.0)
    axes[0].grid(axis='x', alpha=0.3)

    # Figure 2: Cumulative improvement
    config_names = cumulative_df['Configuration'].tolist()
    cum_aucs = [float(x) for x in cumulative_df['ROC-AUC'].tolist()]

    axes[1].plot(range(len(config_names)), cum_aucs, marker='o', linewidth=2, markersize=8)
    axes[1].set_xticks(range(len(config_names)))
    axes[1].set_xticklabels(range(1, len(config_names)+1))
    axes[1].set_xlabel('Number of Models')
    axes[1].set_ylabel('ROC-AUC Score')
    axes[1].set_title('Cumulative Model Addition')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'ablation_study_results.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {viz_path}")

    plt.close()


def main():
    """Run complete ablation study"""
    print("="*80)
    print("ABLATION STUDY: Model Contribution Analysis")
    print("="*80)

    # Setup
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(backend_dir)
    models_dir = os.path.join(project_root, "data", "models")
    output_dir = os.path.join(project_root, "experiments", "results")

    os.makedirs(output_dir, exist_ok=True)

    # Load data and models
    test_df = load_test_data()
    models = load_models(models_dir)

    # Prepare features
    X_test_tabular, X_test_seq, test_df_full, y_test = prepare_test_features(
        test_df, models['preprocessor'], models['seq_preprocessor']
    )

    # Get predictions
    predictions = get_predictions(models, X_test_tabular, X_test_seq, test_df_full)

    # Run ablation studies
    individual_df = ablation_study_individual_models(predictions, y_test)
    cumulative_df = ablation_study_cumulative(predictions, y_test)
    leave_one_out_df = ablation_study_leave_one_out(predictions, y_test)

    # Visualize
    visualize_results(individual_df, cumulative_df, leave_one_out_df, output_dir)

    # Save results
    results_path = os.path.join(output_dir, 'ablation_study_results.json')
    results = {
        'individual_models': individual_df.to_dict('records'),
        'cumulative_addition': cumulative_df.to_dict('records'),
        'leave_one_out': leave_one_out_df.to_dict('records')
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to: {results_path}")

    print("\n" + "="*80)
    print("✅ ABLATION STUDY COMPLETE")
    print("="*80)
    print("\n📊 Key Findings:")
    print(f"  - Number of models tested: {len(predictions)}")
    print(f"  - Best individual model: {individual_df.iloc[individual_df['ROC-AUC'].astype(float).argmax()]['Configuration']}")
    print(f"  - Full ensemble improvement: {cumulative_df.iloc[-1]['Δ AUC']}")
    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
