"""
BERT and FinBERT Models for Financial Fraud Detection
Implements transformer-based models for text-based fraud detection
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import os
import joblib


class FinBERTFraudDetector:
    """
    FinBERT-based fraud detection model
    Uses ProsusAI/finbert or yiyanghkust/finbert-tone for financial text understanding
    """

    def __init__(self, model_name: str = 'ProsusAI/finbert', max_length: int = 512):
        """
        Initialize FinBERT model

        Args:
            model_name: Pretrained model name from HuggingFace
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build FinBERT model for binary classification"""
        # Load pretrained model and add classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,  # Binary classification: fraud vs non-fraud
            output_attentions=True,  # For explainability
            output_hidden_states=True
        )
        self.model.to(self.device)

    def tokenize_texts(self, texts: List[str]) -> Dict:
        """
        Tokenize transaction texts

        Args:
            texts: List of transaction descriptions

        Returns:
            Tokenized inputs
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encodings

    def fit(self, texts: List[str], labels: np.ndarray,
            epochs: int = 3, batch_size: int = 16,
            learning_rate: float = 2e-5, validation_split: float = 0.15):
        """
        Fine-tune FinBERT on fraud detection task

        Args:
            texts: Transaction text descriptions
            labels: Binary labels (0: normal, 1: fraud)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for fine-tuning
            validation_split: Fraction for validation

        Returns:
            Training history
        """
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = self._create_dataset(train_texts, train_labels)
        val_dataset = self._create_dataset(val_texts, val_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./finbert_checkpoints',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            gradient_accumulation_steps=2,  # For larger effective batch size
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )

        # Train
        print("Fine-tuning FinBERT on fraud detection task...")
        trainer.train()

        return trainer

    def _create_dataset(self, texts: List[str], labels: np.ndarray):
        """Create PyTorch dataset from texts and labels"""
        encodings = self.tokenize_texts(texts)

        class TransactionDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

            def __len__(self):
                return len(self.labels)

        return TransactionDataset(encodings, labels)

    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

        predictions, labels = eval_pred
        predictions = np.argmax(predictions[0], axis=1)  # predictions is tuple (logits, ...)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        # For AUC, we need probabilities
        probs = torch.softmax(torch.tensor(eval_pred[0][0]), dim=1)[:, 1].numpy()
        auc = roc_auc_score(labels, probs)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict fraud probability for transactions

        Args:
            texts: Transaction text descriptions

        Returns:
            Fraud probabilities
        """
        self.model.eval()

        # Tokenize
        encodings = self.tokenize_texts(texts)
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)
        fraud_probs = probs[:, 1].cpu().numpy()  # Probability of fraud class

        return fraud_probs

    def get_attention_weights(self, texts: List[str]) -> Dict:
        """
        Extract attention weights for explainability

        Args:
            texts: Transaction text descriptions

        Returns:
            Dictionary with attention weights and token information
        """
        self.model.eval()

        # Tokenize
        encodings = self.tokenize_texts(texts)
        encodings_gpu = {k: v.to(self.device) for k, v in encodings.items()}

        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**encodings_gpu)

        # Extract attention weights (from last layer)
        attentions = outputs.attentions  # Tuple of attention tensors
        last_layer_attention = attentions[-1]  # (batch, heads, seq_len, seq_len)

        # Average across attention heads
        avg_attention = last_layer_attention.mean(dim=1).cpu().numpy()  # (batch, seq_len, seq_len)

        # Get tokens
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in encodings['input_ids']]

        return {
            'attention_weights': avg_attention,
            'tokens': tokens,
            'token_ids': encodings['input_ids'].cpu().numpy(),
            'predictions': torch.softmax(outputs.logits, dim=1).cpu().numpy()
        }

    def save(self, path: str):
        """Save FinBERT model"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        joblib.dump({
            'model_name': self.model_name,
            'max_length': self.max_length
        }, os.path.join(path, 'config.pkl'))

    @staticmethod
    def load(path: str):
        """Load FinBERT model"""
        config = joblib.load(os.path.join(path, 'config.pkl'))
        detector = FinBERTFraudDetector(
            model_name=config['model_name'],
            max_length=config['max_length']
        )
        detector.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            output_attentions=True,
            output_hidden_states=True
        )
        detector.model.to(detector.device)
        detector.tokenizer = AutoTokenizer.from_pretrained(path)
        return detector


class BERTFraudDetector:
    """
    Standard BERT-based fraud detection model
    Alternative to FinBERT using general BERT
    """

    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        """
        Initialize BERT model

        Args:
            model_name: Pretrained BERT model name
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=True,
            output_hidden_states=True
        )
        self.model.to(self.device)

    # Same methods as FinBERT (inherits same structure)
    def tokenize_texts(self, texts: List[str]) -> Dict:
        """Tokenize texts using BERT tokenizer"""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict fraud probability"""
        self.model.eval()
        encodings = self.tokenize_texts(texts)
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=1)
        return probs[:, 1].cpu().numpy()

    def get_attention_weights(self, texts: List[str]) -> Dict:
        """Extract attention weights"""
        self.model.eval()
        encodings = self.tokenize_texts(texts)
        encodings_gpu = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings_gpu)

        attentions = outputs.attentions
        last_layer_attention = attentions[-1]
        avg_attention = last_layer_attention.mean(dim=1).cpu().numpy()

        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in encodings['input_ids']]

        return {
            'attention_weights': avg_attention,
            'tokens': tokens,
            'predictions': torch.softmax(outputs.logits, dim=1).cpu().numpy()
        }
