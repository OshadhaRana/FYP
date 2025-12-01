import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected = [
            'step','type','amount','oldbalanceOrg','newbalanceOrig',
            'oldbalanceDest','newbalanceDest'
        ]
        for col in expected:
            if col not in df.columns:
                df[col] = 0.0 if col != 'type' else 'PAYMENT'
        return df[expected]

    def transform(self, df: pd.DataFrame, fit: bool=False) -> pd.DataFrame:
        df = self._ensure_columns(df.copy())
        df['hour'] = df['step'] % 24
        df['day'] = df['step'] // 24
        df['amount_log'] = np.log1p(df['amount'])

        # Label encode type
        if 'type' not in self.label_encoders:
            self.label_encoders['type'] = LabelEncoder()
            df['type_encoded'] = self.label_encoders['type'].fit_transform(df['type'])
        else:
            df['type_encoded'] = self.label_encoders['type'].transform(df['type'])

        feature_cols = [
            'amount','amount_log','oldbalanceOrg','newbalanceOrig',
            'oldbalanceDest','newbalanceDest','hour','day','type_encoded'
        ]
        X = df[feature_cols].values
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X, feature_cols
