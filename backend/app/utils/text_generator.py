"""
Text Generator for Transaction Data
Converts numerical transaction features into natural language descriptions for BERT/FinBERT
"""

import pandas as pd
import numpy as np
from typing import List


class TransactionTextGenerator:
    """
    Generates natural language descriptions of financial transactions
    for BERT-based models to process
    """

    def __init__(self):
        self.transaction_templates = {
            'PAYMENT': 'Customer {orig} made a payment of {amount} dollars to merchant {dest}. Account balance changed from {old_bal} to {new_bal}.',
            'TRANSFER': 'Customer {orig} transferred {amount} dollars to account {dest}. Original balance was {old_bal}, new balance is {new_bal}.',
            'CASH_OUT': 'Customer {orig} withdrew {amount} dollars in cash. Account balance decreased from {old_bal} to {new_bal}.',
            'DEBIT': 'Customer {orig} made a debit transaction of {amount} dollars. Balance went from {old_bal} to {new_bal}.',
            'CASH_IN': 'Customer {orig} deposited {amount} dollars in cash. Account balance increased from {old_bal} to {new_bal}.'
        }

        self.risk_indicators = {
            'large_amount': 'This is an unusually large transaction.',
            'balance_drain': 'This transaction drains the account balance significantly.',
            'round_amount': 'The transaction amount is a round number, which may indicate suspicious activity.',
            'new_account': 'This is a recently created account.',
            'zero_balance': 'The destination account had zero balance before this transaction.',
            'night_transaction': 'This transaction occurred during nighttime hours.',
            'weekend_transaction': 'This transaction occurred on a weekend.'
        }

    def generate_transaction_text(self, transaction: pd.Series) -> str:
        """
        Generate natural language description of a single transaction

        Args:
            transaction: Pandas Series with transaction features

        Returns:
            Natural language description
        """
        # Base transaction description
        txn_type = transaction.get('type', 'PAYMENT')
        template = self.transaction_templates.get(txn_type, self.transaction_templates['PAYMENT'])

        # Format amounts to 2 decimal places
        amount = f"{transaction.get('amount', 0):.2f}"
        old_bal = f"{transaction.get('oldbalanceOrg', 0):.2f}"
        new_bal = f"{transaction.get('newbalanceOrig', 0):.2f}"

        # Anonymize account IDs (use masked versions)
        orig = self._mask_account_id(transaction.get('nameOrig', 'UNKNOWN'))
        dest = self._mask_account_id(transaction.get('nameDest', 'UNKNOWN'))

        # Fill template
        base_text = template.format(
            orig=orig,
            dest=dest,
            amount=amount,
            old_bal=old_bal,
            new_bal=new_bal
        )

        # Add contextual risk indicators
        risk_text = self._generate_risk_indicators(transaction)

        # Combine
        full_text = f"{base_text} {risk_text}".strip()

        return full_text

    def generate_batch_texts(self, df: pd.DataFrame) -> List[str]:
        """
        Generate text descriptions for multiple transactions

        Args:
            df: DataFrame with transactions

        Returns:
            List of text descriptions
        """
        return [self.generate_transaction_text(row) for _, row in df.iterrows()]

    def _mask_account_id(self, account_id: str) -> str:
        """Mask account ID for privacy while preserving pattern"""
        if pd.isna(account_id) or account_id == 'UNKNOWN':
            return 'UNKNOWN'

        # Extract account type (C for customer, M for merchant)
        if len(account_id) > 0:
            account_type = account_id[0]
            # Mask digits but keep first/last 2 for pattern recognition
            if len(account_id) > 4:
                return f"{account_type}***{account_id[-2:]}"
            return f"{account_type}***"
        return 'UNKNOWN'

    def _generate_risk_indicators(self, transaction: pd.Series) -> str:
        """
        Generate contextual risk indicator text based on transaction features

        Args:
            transaction: Transaction data

        Returns:
            Risk indicator text
        """
        indicators = []

        amount = transaction.get('amount', 0)
        old_bal = transaction.get('oldbalanceOrg', 0)
        new_bal = transaction.get('newbalanceOrig', 0)
        old_bal_dest = transaction.get('oldbalanceDest', 0)

        # Check for large amounts (over 100k)
        if amount > 100000:
            indicators.append(self.risk_indicators['large_amount'])

        # Check for balance drain (>80% of balance)
        if old_bal > 0 and (old_bal - new_bal) / old_bal > 0.8:
            indicators.append(self.risk_indicators['balance_drain'])

        # Check for round amounts (multiples of 1000)
        if amount % 1000 == 0 and amount > 0:
            indicators.append(self.risk_indicators['round_amount'])

        # Check for zero balance destination
        if old_bal_dest == 0:
            indicators.append(self.risk_indicators['zero_balance'])

        # Temporal indicators (if available)
        if 'hour' in transaction:
            hour = transaction['hour']
            if hour >= 22 or hour <= 6:
                indicators.append(self.risk_indicators['night_transaction'])

        if 'is_weekend' in transaction and transaction['is_weekend']:
            indicators.append(self.risk_indicators['weekend_transaction'])

        return ' '.join(indicators)

    def generate_sequence_narrative(self, transactions_df: pd.DataFrame) -> str:
        """
        Generate a narrative describing a sequence of transactions (for LSTM+BERT hybrid)

        Args:
            transactions_df: DataFrame with sequence of transactions for one user

        Returns:
            Narrative text describing the transaction sequence
        """
        if len(transactions_df) == 0:
            return "No transaction history available."

        # Sort by step/timestamp
        transactions_df = transactions_df.sort_values('step')

        # Get summary statistics
        total_amount = transactions_df['amount'].sum()
        avg_amount = transactions_df['amount'].mean()
        num_transactions = len(transactions_df)

        # Build narrative
        narrative = f"Account has {num_transactions} recent transactions with total volume of ${total_amount:.2f}. "
        narrative += f"Average transaction amount is ${avg_amount:.2f}. "

        # Describe pattern
        if transactions_df['amount'].std() > avg_amount * 0.5:
            narrative += "Transaction amounts show high variability. "
        else:
            narrative += "Transaction amounts are relatively consistent. "

        # Describe most recent transaction
        latest = transactions_df.iloc[-1]
        narrative += f"Most recent transaction: {self.generate_transaction_text(latest)}"

        return narrative
