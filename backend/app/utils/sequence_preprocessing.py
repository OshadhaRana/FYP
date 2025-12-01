"""
Sequence Preprocessing for LSTM Model
Converts transaction history into sequential format for temporal pattern analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from collections import defaultdict


class SequencePreprocessor:
    """
    Converts individual transactions into user-level sequences for LSTM processing.
    Groups transactions by user (nameOrig) and creates fixed-length sequences.
    """

    def __init__(self, sequence_length: int = 10, stride: int = 1):
        """
        Initialize sequence preprocessor

        Args:
            sequence_length: Number of transactions per sequence
            stride: Step size for sliding window
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.user_histories = defaultdict(list)

    def create_sequences(self, df: pd.DataFrame, feature_cols: List[str],
                        label_col: str = 'isFraud') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequential data from transaction DataFrame

        Args:
            df: DataFrame with transactions
            feature_cols: List of feature column names
            label_col: Name of label column

        Returns:
            X_seq: Sequential features (n_sequences, sequence_length, n_features)
            y_seq: Labels for sequences (n_sequences,)
        """
        # Sort by user and timestamp
        df = df.sort_values(['nameOrig', 'step']).reset_index(drop=True)

        sequences = []
        labels = []

        # Group by user
        for user_id, group in df.groupby('nameOrig'):
            # Skip users with insufficient transaction history
            if len(group) < self.sequence_length:
                # Pad if needed
                n_padding = self.sequence_length - len(group)
                user_features = group[feature_cols].values
                user_labels = group[label_col].values

                # Pad with zeros at the beginning
                padded_features = np.vstack([
                    np.zeros((n_padding, len(feature_cols))),
                    user_features
                ])

                sequences.append(padded_features)
                # Label is 1 if ANY transaction in sequence is fraud
                labels.append(int(np.any(user_labels)))
            else:
                # Create sliding windows
                user_features = group[feature_cols].values
                user_labels = group[label_col].values

                for i in range(0, len(group) - self.sequence_length + 1, self.stride):
                    seq_features = user_features[i:i + self.sequence_length]
                    seq_labels = user_labels[i:i + self.sequence_length]

                    sequences.append(seq_features)
                    # Label is 1 if ANY transaction in sequence is fraud
                    labels.append(int(np.any(seq_labels)))

        X_seq = np.array(sequences)
        y_seq = np.array(labels)

        return X_seq, y_seq

    def create_sequence_for_single_transaction(self, transaction: pd.DataFrame,
                                               user_history: pd.DataFrame,
                                               feature_cols: List[str]) -> np.ndarray:
        """
        Create sequence for a single transaction using user history

        Args:
            transaction: Current transaction (single row DataFrame)
            user_history: Historical transactions for this user
            feature_cols: Feature columns to use

        Returns:
            Sequence array (1, sequence_length, n_features)
        """
        # Combine history with current transaction
        all_transactions = pd.concat([user_history, transaction], ignore_index=True)
        all_transactions = all_transactions.sort_values('step').reset_index(drop=True)

        # Take last sequence_length transactions
        recent_transactions = all_transactions.tail(self.sequence_length)

        # Pad if insufficient history
        if len(recent_transactions) < self.sequence_length:
            n_padding = self.sequence_length - len(recent_transactions)
            features = recent_transactions[feature_cols].values
            padded_features = np.vstack([
                np.zeros((n_padding, len(feature_cols))),
                features
            ])
        else:
            padded_features = recent_transactions[feature_cols].values

        return padded_features.reshape(1, self.sequence_length, len(feature_cols))

    def create_sequences_from_processed(self, X_processed: np.ndarray,
                                       df_original: pd.DataFrame,
                                       label_col: str = 'isFraud') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from already processed features

        Args:
            X_processed: Processed feature array (n_samples, n_features)
            df_original: Original DataFrame with user IDs and timestamps
            label_col: Label column name

        Returns:
            X_seq: Sequential features
            y_seq: Sequence labels
        """
        # Create temporary DataFrame
        temp_df = df_original[['nameOrig', 'step', label_col]].copy()
        temp_df['features'] = list(X_processed)

        sequences = []
        labels = []

        # Group by user
        for user_id, group in temp_df.groupby('nameOrig'):
            if len(group) < self.sequence_length:
                # Pad short sequences
                n_padding = self.sequence_length - len(group)
                user_features = np.vstack(group['features'].values)
                user_labels = group[label_col].values

                padded_features = np.vstack([
                    np.zeros((n_padding, user_features.shape[1])),
                    user_features
                ])

                sequences.append(padded_features)
                labels.append(int(np.any(user_labels)))
            else:
                # Create sliding windows
                user_features = np.vstack(group['features'].values)
                user_labels = group[label_col].values

                for i in range(0, len(group) - self.sequence_length + 1, self.stride):
                    seq_features = user_features[i:i + self.sequence_length]
                    seq_labels = user_labels[i:i + self.sequence_length]

                    sequences.append(seq_features)
                    labels.append(int(np.any(seq_labels)))

        X_seq = np.array(sequences)
        y_seq = np.array(labels)

        return X_seq, y_seq

    def get_sequence_stats(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about sequence generation

        Args:
            df: Transaction DataFrame

        Returns:
            Dictionary with statistics
        """
        user_counts = df.groupby('nameOrig').size()

        stats = {
            'total_users': len(user_counts),
            'users_with_sufficient_history': (user_counts >= self.sequence_length).sum(),
            'users_requiring_padding': (user_counts < self.sequence_length).sum(),
            'avg_transactions_per_user': user_counts.mean(),
            'max_transactions_per_user': user_counts.max(),
            'min_transactions_per_user': user_counts.min()
        }

        return stats


class TemporalFeatureEngineer:
    """
    Additional temporal feature engineering for sequential models
    """

    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for sequence analysis

        Args:
            df: Transaction DataFrame

        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()

        # Time-based features
        df['hour_of_day'] = df['step'] % 24
        df['day_of_month'] = (df['step'] // 24) % 30
        df['is_weekend'] = ((df['step'] // 24) % 7 >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)

        # Transaction velocity features (requires grouping by user)
        df = df.sort_values(['nameOrig', 'step'])

        # Time since last transaction
        df['time_since_last_txn'] = df.groupby('nameOrig')['step'].diff().fillna(0)

        # Rolling statistics (requires grouping)
        df['rolling_avg_amount'] = df.groupby('nameOrig')['amount'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )

        df['rolling_std_amount'] = df.groupby('nameOrig')['amount'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        ).fillna(0)

        # Amount deviation from user average
        df['amount_deviation'] = df['amount'] - df['rolling_avg_amount']

        # Balance change rate
        df['balance_change_rate'] = (
            (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1)
        )

        return df

    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between temporal and transaction features

        Args:
            df: Transaction DataFrame

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        # Amount-time interactions
        df['amount_x_hour'] = df['amount'] * df['hour_of_day']
        df['amount_x_is_night'] = df['amount'] * df['is_night']

        # Balance-time interactions
        df['balance_ratio'] = (df['oldbalanceOrg'] + 1) / (df['newbalanceOrig'] + 1)

        return df
