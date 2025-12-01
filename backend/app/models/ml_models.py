import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

class FraudDetectionEnsemble:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

    def fit(self, X_df, y):
        X_scaled, _ = self.preprocessor.transform(X_df, fit=True)
        self.rf_model.fit(X_scaled, y)
        self.xgb_model.fit(X_scaled, y)

    def predict_proba(self, X_df):
        X_scaled, _ = self.preprocessor.transform(X_df, fit=False)
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
        return ((rf_proba + xgb_proba) / 2).reshape(-1, 1)

    def save(self, path):
        joblib.dump({
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'preprocessor': self.preprocessor
        }, path)

    @staticmethod
    def load(path):
        bundle = joblib.load(path)
        model = FraudDetectionEnsemble(bundle['preprocessor'])
        model.rf_model = bundle['rf_model']
        model.xgb_model = bundle['xgb_model']
        return model
