"""
Explainability methods for deep learning models (LSTM, CNN)
Implements attention visualization, gradient-based methods, and LIME for neural networks
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple


class DeepLearningExplainer:
    """
    Explainability toolkit for deep learning fraud detection models
    Supports LSTM and CNN interpretability
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer

        Args:
            model: Trained Keras model
            feature_names: Names of input features
        """
        self.model = model
        self.feature_names = feature_names

    def explain_with_gradients(self, X_input: np.ndarray) -> Dict:
        """
        Use gradient-based methods to explain predictions (Integrated Gradients)

        Args:
            X_input: Input data (batch_size, ...)

        Returns:
            Dictionary with gradient-based explanations
        """
        X_tensor = tf.convert_to_tensor(X_input, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)

        # Compute gradients
        gradients = tape.gradient(predictions, X_tensor)

        # Gradient * Input (approximation of integrated gradients)
        attributions = tf.abs(gradients * X_tensor).numpy()

        return {
            'attributions': attributions,
            'gradients': gradients.numpy(),
            'predictions': predictions.numpy()
        }

    def explain_cnn_with_gradcam(self, X_input: np.ndarray,
                                 conv_layer_name: str = 'conv1d_2') -> Dict:
        """
        Grad-CAM for CNN models to visualize important features

        Args:
            X_input: Input data
            conv_layer_name: Name of convolutional layer

        Returns:
            Dictionary with Grad-CAM heatmaps
        """
        # Get model up to conv layer and from conv layer to output
        conv_layer = self.model.get_layer(conv_layer_name)

        grad_model = keras.models.Model(
            inputs=self.model.input,
            outputs=[conv_layer.output, self.model.output]
        )

        X_tensor = tf.convert_to_tensor(X_input, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(X_tensor)
            # Focus on fraud class (positive class)
            class_output = predictions[:, 0]

        # Gradient of class output with respect to conv layer
        grads = tape.gradient(class_output, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        # Weight conv outputs by gradient importance
        conv_outputs = conv_outputs.numpy()[0]
        pooled_grads = pooled_grads.numpy()

        # Weighted combination
        heatmap = np.zeros(conv_outputs.shape[0])
        for i, grad in enumerate(pooled_grads):
            heatmap += grad * conv_outputs[:, i]

        # Normalize
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return {
            'heatmap': heatmap,
            'predictions': predictions.numpy(),
            'feature_importance': dict(zip(self.feature_names, heatmap[:len(self.feature_names)]))
        }

    def explain_lstm_with_attention(self, X_seq: np.ndarray) -> Dict:
        """
        Visualize LSTM attention over time steps

        Note: This requires the model to have explicit attention mechanism.
        For standard LSTM, we use gradient-based approximation.

        Args:
            X_seq: Sequential input (batch_size, sequence_length, n_features)

        Returns:
            Dictionary with time-step importance
        """
        X_tensor = tf.convert_to_tensor(X_seq, dtype=tf.float32)

        # Compute gradients for each time step
        time_step_importance = []

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)

        gradients = tape.gradient(predictions, X_tensor)
        gradients = np.abs(gradients.numpy())

        # Importance per time step (sum over features)
        for t in range(X_seq.shape[1]):
            time_step_importance.append(np.mean(gradients[:, t, :]))

        # Normalize
        time_step_importance = np.array(time_step_importance)
        if time_step_importance.max() > 0:
            time_step_importance /= time_step_importance.max()

        return {
            'time_step_importance': time_step_importance,
            'predictions': predictions.numpy(),
            'detailed_gradients': gradients
        }

    def explain_with_lime(self, X_input: np.ndarray, mode='tabular') -> Dict:
        """
        Use LIME to explain deep learning predictions

        Args:
            X_input: Input data
            mode: 'tabular' for standard features

        Returns:
            LIME explanations
        """
        if mode == 'tabular':
            # Flatten if needed
            if len(X_input.shape) > 2:
                original_shape = X_input.shape
                X_flattened = X_input.reshape(X_input.shape[0], -1)
            else:
                X_flattened = X_input
                original_shape = None

            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_flattened,
                feature_names=[f"feature_{i}" for i in range(X_flattened.shape[1])],
                class_names=['normal', 'fraud'],
                mode='classification'
            )

            # Explain first instance
            def predict_fn(X):
                if original_shape is not None:
                    X = X.reshape(-1, *original_shape[1:])
                return self.model.predict(X, verbose=0)

            explanation = explainer.explain_instance(
                X_flattened[0],
                predict_fn,
                num_features=min(10, X_flattened.shape[1])
            )

            return {
                'lime_explanation': explanation,
                'feature_importance': dict(explanation.as_list())
            }

    def explain_with_shap_deep(self, X_background: np.ndarray,
                               X_input: np.ndarray) -> Dict:
        """
        Use SHAP DeepExplainer for neural networks

        Args:
            X_background: Background dataset for SHAP
            X_input: Input to explain

        Returns:
            SHAP values and explanations
        """
        # Create SHAP Deep Explainer
        explainer = shap.DeepExplainer(self.model, X_background)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_input)

        # If sequence data, aggregate over time
        if len(shap_values[0].shape) > 2:
            # Average over sequence dimension
            shap_values_aggregated = np.mean(shap_values[0], axis=1)
        else:
            shap_values_aggregated = shap_values[0]

        # Create feature importance dictionary
        feature_importance = {}
        for i, fname in enumerate(self.feature_names[:shap_values_aggregated.shape[1]]):
            feature_importance[fname] = float(np.mean(np.abs(shap_values_aggregated[:, i])))

        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'expected_value': explainer.expected_value
        }


class EnsembleExplainer:
    """
    Unified explainer for multi-model ensemble
    Combines explanations from multiple models
    """

    def __init__(self, models: Dict, feature_names: List[str]):
        """
        Initialize ensemble explainer

        Args:
            models: Dictionary of models {'rf': rf_model, 'xgb': xgb_model, etc.}
            feature_names: Feature names
        """
        self.models = models
        self.feature_names = feature_names
        self.dl_explainers = {}

        # Create explainers for deep learning models
        for name, model in models.items():
            if name in ['lstm', 'cnn']:
                self.dl_explainers[name] = DeepLearningExplainer(model, feature_names)

    def explain_ensemble_prediction(self, X_tabular: np.ndarray,
                                   X_seq: np.ndarray = None,
                                   ensemble_weights: Dict = None) -> Dict:
        """
        Generate comprehensive explanations from all models

        Args:
            X_tabular: Tabular features
            X_seq: Sequential features (for LSTM)
            ensemble_weights: Model weights in ensemble

        Returns:
            Unified explanation dictionary
        """
        explanations = {
            'model_predictions': {},
            'feature_importance': {},
            'ensemble_contribution': {}
        }

        # Get predictions from each model
        for model_name, model in self.models.items():
            if model_name == 'lstm' and X_seq is not None:
                pred = model.predict_proba(X_seq)
            elif model_name in ['rf', 'xgb', 'cnn']:
                pred = model.predict_proba(X_tabular)
            else:
                continue

            explanations['model_predictions'][model_name] = float(pred[0] if len(pred.shape) > 1 else pred)

        # Feature importance from tree models (SHAP)
        if 'rf' in self.models or 'xgb' in self.models:
            explanations['feature_importance']['traditional_ml'] = self._explain_tree_models(X_tabular)

        # Feature importance from CNN
        if 'cnn' in self.dl_explainers:
            cnn_explanation = self.dl_explainers['cnn'].explain_with_gradients(X_tabular)
            explanations['feature_importance']['cnn'] = cnn_explanation

        # Time-step importance from LSTM
        if 'lstm' in self.dl_explainers and X_seq is not None:
            lstm_explanation = self.dl_explainers['lstm'].explain_lstm_with_attention(X_seq)
            explanations['feature_importance']['lstm'] = lstm_explanation

        # Calculate weighted contributions
        if ensemble_weights:
            for model_name, weight in ensemble_weights.items():
                if model_name in explanations['model_predictions']:
                    explanations['ensemble_contribution'][model_name] = (
                        explanations['model_predictions'][model_name] * weight
                    )

        return explanations

    def _explain_tree_models(self, X_tabular: np.ndarray) -> Dict:
        """
        Generate SHAP explanations for tree-based models

        Args:
            X_tabular: Input features

        Returns:
            SHAP feature importance
        """
        feature_importance = {}

        for model_name in ['rf', 'xgb']:
            if model_name in self.models:
                explainer = shap.TreeExplainer(self.models[model_name])
                shap_values = explainer.shap_values(X_tabular)

                # Handle binary classification
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                # Aggregate importance
                importance = np.abs(shap_values).mean(axis=0)
                feature_importance[model_name] = dict(zip(self.feature_names, importance))

        return feature_importance

    def format_explanation_for_stakeholder(self, explanations: Dict,
                                          stakeholder_type: str) -> Dict:
        """
        Format explanations for specific stakeholder (compliance officer vs risk analyst)

        Args:
            explanations: Raw explanations
            stakeholder_type: 'compliance' or 'risk_analyst'

        Returns:
            Formatted explanation
        """
        if stakeholder_type == 'compliance':
            return {
                'risk_level': self._categorize_risk(explanations),
                'model_consensus': self._check_model_consensus(explanations),
                'regulatory_notes': self._generate_regulatory_notes(explanations),
                'audit_trail': {
                    'models_used': list(explanations['model_predictions'].keys()),
                    'timestamp': 'generated_timestamp',
                    'prediction_details': explanations['model_predictions']
                }
            }
        else:  # risk_analyst
            return {
                'detailed_predictions': explanations['model_predictions'],
                'feature_importance': explanations['feature_importance'],
                'model_contributions': explanations.get('ensemble_contribution', {}),
                'statistical_confidence': self._compute_confidence_metrics(explanations)
            }

    def _categorize_risk(self, explanations: Dict) -> str:
        """Categorize overall risk level"""
        avg_pred = np.mean(list(explanations['model_predictions'].values()))
        if avg_pred > 0.7:
            return 'HIGH'
        elif avg_pred > 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _check_model_consensus(self, explanations: Dict) -> str:
        """Check if models agree on prediction"""
        preds = list(explanations['model_predictions'].values())
        std = np.std(preds)
        if std < 0.1:
            return 'Strong consensus'
        elif std < 0.2:
            return 'Moderate consensus'
        else:
            return 'Weak consensus - investigate further'

    def _generate_regulatory_notes(self, explanations: Dict) -> str:
        """Generate compliance-oriented notes"""
        avg_pred = np.mean(list(explanations['model_predictions'].values()))
        if avg_pred > 0.7:
            return 'HIGH RISK: Requires immediate AML review and SAR filing consideration'
        elif avg_pred > 0.3:
            return 'MEDIUM RISK: Enhanced due diligence recommended'
        else:
            return 'LOW RISK: Standard monitoring procedures apply'

    def _compute_confidence_metrics(self, explanations: Dict) -> Dict:
        """Compute statistical confidence metrics"""
        preds = list(explanations['model_predictions'].values())
        return {
            'mean_prediction': float(np.mean(preds)),
            'std_prediction': float(np.std(preds)),
            'min_prediction': float(np.min(preds)),
            'max_prediction': float(np.max(preds))
        }
