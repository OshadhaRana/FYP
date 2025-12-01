"""
Deep Learning Models for Financial Crime Detection
Implements LSTM, CNN, and ensemble architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from typing import Tuple, Dict, List


class LSTMFraudDetector:
    """
    LSTM model for detecting fraud patterns in sequential transaction data.
    Analyzes user transaction history to identify suspicious behavioral patterns.
    """

    def __init__(self, sequence_length: int = 10, n_features: int = 9):
        """
        Initialize LSTM Fraud Detector

        Args:
            sequence_length: Number of historical transactions to consider
            n_features: Number of features per transaction
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build LSTM architecture optimized for fraud detection"""
        self.model = models.Sequential([
            # LSTM Layer 1: Learn temporal patterns
            layers.LSTM(64, return_sequences=True,
                       input_shape=(self.sequence_length, self.n_features),
                       name='lstm_1'),
            layers.Dropout(0.3, name='dropout_1'),

            # LSTM Layer 2: Capture complex sequential dependencies
            layers.LSTM(32, return_sequences=False, name='lstm_2'),
            layers.Dropout(0.3, name='dropout_2'),

            # Dense layers for classification
            layers.Dense(16, activation='relu', name='dense_1'),
            layers.Dropout(0.2, name='dropout_3'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall',
                    keras.metrics.AUC(name='auc')]
        )

    def fit(self, X_seq, y, validation_split=0.2, epochs=50, batch_size=32, verbose=1):
        """
        Train LSTM model on sequential transaction data

        Args:
            X_seq: Sequential transaction data (n_samples, sequence_length, n_features)
            y: Labels (n_samples,)
            validation_split: Fraction of data for validation
            epochs: Maximum training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode

        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5,
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                             patience=3, min_lr=0.00001, verbose=1)
        ]

        history = self.model.fit(
            X_seq, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=self._compute_class_weights(y)
        )

        return history

    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

    def predict_proba(self, X_seq):
        """
        Predict fraud probability for sequences

        Args:
            X_seq: Sequential transaction data

        Returns:
            Fraud probabilities
        """
        return self.model.predict(X_seq, verbose=0)

    def get_attention_weights(self, X_seq):
        """
        Extract attention weights for explainability
        Note: This requires modifying the model architecture to include attention
        """
        # Placeholder for attention mechanism
        # Would require implementing attention layer in model
        return None

    def save(self, path):
        """Save LSTM model"""
        self.model.save(f"{path}_lstm.h5")
        joblib.dump({
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }, f"{path}_lstm_config.pkl")

    @staticmethod
    def load(path):
        """Load LSTM model"""
        config = joblib.load(f"{path}_lstm_config.pkl")
        model = LSTMFraudDetector(
            sequence_length=config['sequence_length'],
            n_features=config['n_features']
        )
        model.model = keras.models.load_model(f"{path}_lstm.h5")
        return model


class CNN1DFraudDetector:
    """
    1D CNN model for feature extraction and pattern recognition in transactions.
    Treats transaction features as 1D signals to detect local patterns.
    """

    def __init__(self, n_features: int = 9):
        """
        Initialize 1D CNN Fraud Detector

        Args:
            n_features: Number of input features
        """
        self.n_features = n_features
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build 1D CNN architecture for fraud detection"""
        self.model = models.Sequential([
            # Reshape for 1D CNN (treat features as sequence)
            layers.Reshape((self.n_features, 1),
                          input_shape=(self.n_features,),
                          name='reshape'),

            # Conv Block 1: Detect local patterns
            layers.Conv1D(64, kernel_size=3, activation='relu',
                         padding='same', name='conv1d_1'),
            layers.BatchNormalization(name='bn_1'),
            layers.MaxPooling1D(pool_size=2, name='pool_1'),
            layers.Dropout(0.3, name='dropout_1'),

            # Conv Block 2: Learn hierarchical features
            layers.Conv1D(32, kernel_size=3, activation='relu',
                         padding='same', name='conv1d_2'),
            layers.BatchNormalization(name='bn_2'),
            layers.GlobalMaxPooling1D(name='global_pool'),

            # Dense layers for classification
            layers.Dense(16, activation='relu', name='dense_1'),
            layers.Dropout(0.3, name='dropout_2'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall',
                    keras.metrics.AUC(name='auc')]
        )

    def fit(self, X, y, validation_split=0.2, epochs=50, batch_size=32, verbose=1):
        """
        Train CNN model

        Args:
            X: Transaction features (n_samples, n_features)
            y: Labels (n_samples,)
            validation_split: Fraction for validation
            epochs: Maximum training epochs
            batch_size: Batch size
            verbose: Verbosity mode

        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5,
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                             patience=3, min_lr=0.00001, verbose=1)
        ]

        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=self._compute_class_weights(y)
        )

        return history

    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

    def predict_proba(self, X):
        """
        Predict fraud probability

        Args:
            X: Transaction features

        Returns:
            Fraud probabilities
        """
        return self.model.predict(X, verbose=0)

    def get_feature_maps(self, X):
        """Extract intermediate feature maps for explainability"""
        feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('conv1d_2').output
        )
        return feature_extractor.predict(X, verbose=0)

    def save(self, path):
        """Save CNN model"""
        self.model.save(f"{path}_cnn.h5")
        joblib.dump({'n_features': self.n_features}, f"{path}_cnn_config.pkl")

    @staticmethod
    def load(path):
        """Load CNN model"""
        config = joblib.load(f"{path}_cnn_config.pkl")
        model = CNN1DFraudDetector(n_features=config['n_features'])
        model.model = keras.models.load_model(f"{path}_cnn.h5")
        return model


class MultiModelEnsemble:
    """
    Meta-learner ensemble combining RF, XGBoost, LSTM, and CNN predictions.
    Uses weighted averaging or stacking for final predictions.
    """

    def __init__(self, rf_model, xgb_model, lstm_model, cnn_model):
        """
        Initialize ensemble with trained models

        Args:
            rf_model: Trained Random Forest model
            xgb_model: Trained XGBoost model
            lstm_model: Trained LSTM model
            cnn_model: Trained CNN model
        """
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model

        # Default weights (can be optimized via validation)
        self.weights = {
            'rf': 0.25,
            'xgb': 0.30,
            'lstm': 0.25,
            'cnn': 0.20
        }

    def predict_proba(self, X_tabular, X_seq=None):
        """
        Generate ensemble predictions

        Args:
            X_tabular: Tabular features for RF, XGBoost, CNN
            X_seq: Sequential features for LSTM (optional)

        Returns:
            Ensemble fraud probabilities
        """
        predictions = {}

        # Traditional ML models
        predictions['rf'] = self.rf_model.predict_proba(X_tabular)[:, 1]
        predictions['xgb'] = self.xgb_model.predict_proba(X_tabular)[:, 1]

        # CNN predictions
        predictions['cnn'] = self.cnn_model.predict_proba(X_tabular).flatten()

        # LSTM predictions (if sequence data available)
        if X_seq is not None:
            predictions['lstm'] = self.lstm_model.predict_proba(X_seq).flatten()
        else:
            # Use average of other models if no sequence data
            predictions['lstm'] = (predictions['rf'] + predictions['xgb'] +
                                  predictions['cnn']) / 3

        # Weighted average ensemble
        ensemble_proba = sum(predictions[model] * self.weights[model]
                            for model in self.weights.keys())

        return ensemble_proba, predictions

    def optimize_weights(self, X_tabular, X_seq, y_true):
        """
        Optimize ensemble weights on validation data

        Args:
            X_tabular: Validation tabular features
            X_seq: Validation sequential features
            y_true: True labels

        Returns:
            Optimized weights
        """
        from sklearn.metrics import roc_auc_score
        from scipy.optimize import minimize

        def objective(weights):
            """Minimize negative AUC"""
            self.weights = {
                'rf': weights[0],
                'xgb': weights[1],
                'lstm': weights[2],
                'cnn': weights[3]
            }
            proba, _ = self.predict_proba(X_tabular, X_seq)
            return -roc_auc_score(y_true, proba)

        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * 4
        initial_weights = [0.25, 0.30, 0.25, 0.20]

        result = minimize(objective, initial_weights,
                         method='SLSQP', bounds=bounds,
                         constraints=constraints)

        self.weights = {
            'rf': result.x[0],
            'xgb': result.x[1],
            'lstm': result.x[2],
            'cnn': result.x[3]
        }

        return self.weights

    def get_model_contributions(self, X_tabular, X_seq=None):
        """
        Get individual model predictions for explainability

        Returns:
            Dictionary of model predictions and contributions
        """
        ensemble_proba, predictions = self.predict_proba(X_tabular, X_seq)

        contributions = {
            model: predictions[model] * self.weights[model]
            for model in self.weights.keys()
        }

        return {
            'ensemble_prediction': ensemble_proba,
            'individual_predictions': predictions,
            'weighted_contributions': contributions,
            'weights': self.weights
        }
