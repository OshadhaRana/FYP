import numpy as np
import shap

class FinCrimeShapExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = None
        self.expected_value_ = None

    def fit(self, background_array: np.ndarray):
        # KernelExplainer for model-agnostic predictions
        self.explainer = shap.KernelExplainer(
            lambda X: self.model.predict_proba_from_array(X),
            background_array
        )
        # Estimate expected value with the background
        self.expected_value_ = self.explainer.expected_value

    def shap_values_for_array(self, sample_array: np.ndarray):
        vals = self.explainer.shap_values(sample_array, nsamples=100)
        # Binary classification -> pick the positive class if returned as list
        if isinstance(vals, list) and len(vals) > 1:
            return vals[1]
        return vals
