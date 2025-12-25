"""
Graph Neural Networks for Fraud Detection
Detects fraud patterns in transaction networks using graph-based deep learning

Research Contribution: Models transactions as graphs to capture:
- Money laundering patterns (circular flows)
- Fraud rings (connected fraudulent accounts)
- Entity relationships (account-to-account networks)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import pandas as pd
import networkx as nx
import joblib


class TemporalGraphBuilder:
    """
    Builds transaction graphs from tabular transaction data

    Graph Structure:
    - Nodes: Accounts (both origin and destination)
    - Edges: Transactions (directed, with features)
    - Node features: Account statistics, transaction history
    - Edge features: Amount, type, temporal info
    """

    def __init__(self, time_window: int = 24):
        """
        Initialize graph builder

        Args:
            time_window: Time window (in hours) to consider for graph construction
        """
        self.time_window = time_window
        self.account_to_idx = {}
        self.idx_to_account = {}

    def build_graph_from_transactions(self, df: pd.DataFrame) -> Data:
        """
        Build PyTorch Geometric graph from transaction DataFrame

        Args:
            df: Transaction DataFrame

        Returns:
            PyTorch Geometric Data object
        """
        # Create account index mapping
        unique_accounts = set(df['nameOrig'].unique()) | set(df['nameDest'].unique())
        self.account_to_idx = {acc: idx for idx, acc in enumerate(unique_accounts)}
        self.idx_to_account = {idx: acc for acc, idx in self.account_to_idx.items()}

        num_nodes = len(unique_accounts)

        # Build node features (account-level statistics)
        node_features = self._compute_node_features(df, num_nodes)

        # Build edge index and edge features
        edge_index, edge_features, edge_labels = self._compute_edges(df)

        # Create graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=edge_labels,
            num_nodes=num_nodes
        )

        return graph

    def _compute_node_features(self, df: pd.DataFrame, num_nodes: int) -> torch.Tensor:
        """
        Compute node (account) features

        Features include:
        - Total transaction volume (sent/received)
        - Number of transactions
        - Average transaction amount
        - Account activity pattern

        FIXED: Removed fraud rate to prevent target leakage
        """
        node_features = np.zeros((num_nodes, 7))  # 7 features per node (was 8)

        # Group by origin account - REMOVED isFraud aggregation
        origin_stats = df.groupby('nameOrig').agg({
            'amount': ['sum', 'mean', 'count'],
            'step': ['min', 'max']
        })

        # Group by destination account
        dest_stats = df.groupby('nameDest').agg({
            'amount': ['sum', 'mean', 'count']
        })

        # Fill node features for origin accounts
        for account, idx in self.account_to_idx.items():
            if account in origin_stats.index:
                stats = origin_stats.loc[account]
                node_features[idx, 0] = stats[('amount', 'sum')]  # Total sent
                node_features[idx, 1] = stats[('amount', 'mean')]  # Avg sent
                node_features[idx, 2] = stats[('amount', 'count')]  # Num sent
                node_features[idx, 5] = stats[('step', 'max')] - stats[('step', 'min')]  # Activity span

            if account in dest_stats.index:
                stats = dest_stats.loc[account]
                node_features[idx, 3] = stats[('amount', 'sum')]  # Total received
                node_features[idx, 4] = stats[('amount', 'mean')]  # Avg received
                node_features[idx, 6] = stats[('amount', 'count')]  # Num received

        # Normalize features
        node_features = torch.FloatTensor(node_features)
        node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-8)

        return node_features

    def _compute_edges(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute edges (transactions) and their features

        Edge features:
        - Transaction amount (normalized)
        - Transaction type (one-hot)
        - Time of day
        - Balance changes
        """
        edge_index = []
        edge_features = []
        edge_labels = []

        for _, row in df.iterrows():
            # Get node indices
            src_idx = self.account_to_idx[row['nameOrig']]
            dst_idx = self.account_to_idx[row['nameDest']]

            # Edge index (directed graph)
            edge_index.append([src_idx, dst_idx])

            # Edge features (7 features) - FIXED: Removed data leakage
            # Removed balance change rate that used newbalanceOrig (future information)
            features = [
                np.log1p(row['amount']),  # Log amount
                row.get('hour', row['step'] % 24) / 24.0,  # Normalized hour
                row['oldbalanceOrg'] / (row['oldbalanceOrg'] + 1),  # Balance utilization (before txn)
                row['oldbalanceDest'] / (row['oldbalanceDest'] + 1),  # Dest balance utilization (before txn)
                1 if row['type'] == 'TRANSFER' else 0,  # Type: TRANSFER
                1 if row['type'] == 'CASH_OUT' else 0,  # Type: CASH_OUT
                1 if row['type'] == 'PAYMENT' else 0,  # Type: PAYMENT
            ]
            edge_features.append(features)

            # Edge label (fraud or not)
            edge_labels.append(row['isFraud'])

        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_features = torch.FloatTensor(edge_features)
        edge_labels = torch.LongTensor(edge_labels)

        return edge_index, edge_features, edge_labels

    def detect_fraud_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect specific fraud patterns in transaction graph

        Patterns:
        - Circular flows (A→B→C→A)
        - Fan-out (one account to many)
        - Rapid sequence (multiple transactions in short time)
        """
        G = nx.DiGraph()

        # Build NetworkX graph for pattern analysis
        for _, row in df.iterrows():
            G.add_edge(
                row['nameOrig'],
                row['nameDest'],
                amount=row['amount'],
                time=row['step'],
                fraud=row['isFraud']
            )

        patterns = {
            'circular_flows': self._find_circular_flows(G),
            'fan_out_accounts': self._find_fan_out(G),
            'rapid_sequences': self._find_rapid_sequences(df)
        }

        return patterns

    def _find_circular_flows(self, G: nx.DiGraph) -> List[List[str]]:
        """Find circular transaction patterns (money laundering indicator)"""
        try:
            cycles = list(nx.simple_cycles(G))
            # Filter for cycles of length 3-5 (most suspicious)
            return [cycle for cycle in cycles if 3 <= len(cycle) <= 5][:10]  # Top 10
        except:
            return []

    def _find_fan_out(self, G: nx.DiGraph, threshold: int = 5) -> List[str]:
        """Find accounts sending to many destinations (fraud distribution)"""
        fan_out_accounts = []
        for node in G.nodes():
            out_degree = G.out_degree(node)
            if out_degree >= threshold:
                fan_out_accounts.append(node)
        return fan_out_accounts[:10]  # Top 10

    def _find_rapid_sequences(self, df: pd.DataFrame, time_threshold: int = 1) -> List[str]:
        """Find accounts with rapid transaction sequences"""
        rapid_accounts = []

        for account in df['nameOrig'].unique():
            account_txns = df[df['nameOrig'] == account].sort_values('step')
            if len(account_txns) > 1:
                time_diffs = account_txns['step'].diff()
                if (time_diffs < time_threshold).sum() > 3:  # 3+ transactions within threshold
                    rapid_accounts.append(account)

        return rapid_accounts[:10]  # Top 10


class GNNFraudDetector(nn.Module):
    """
    Graph Neural Network for fraud detection

    Architecture:
    - Graph Attention Network (GAT) layers for neighborhood aggregation
    - Edge-level prediction (transaction fraud detection)
    - Attention mechanism for explainability
    """

    def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 64, num_layers: int = 3):
        """
        Initialize GNN model

        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
        """
        super(GNNFraudDetector, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Node embedding
        self.node_encoder = nn.Linear(node_features, hidden_dim)

        # Edge embedding
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        # GAT layers for node updates
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Edge classifier (fraud prediction)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_features, hidden_dim),  # src + dst + edge
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Binary classification
        )

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        """
        Forward pass

        Args:
            x: Node features (num_nodes, node_features)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_features)
            return_attention: Whether to return attention weights

        Returns:
            Edge-level predictions
        """
        # Encode nodes
        x = F.relu(self.node_encoder(x))

        # Encode edges
        edge_embeddings = F.relu(self.edge_encoder(edge_attr))

        # Apply GAT layers
        attention_weights = []
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            if return_attention:
                x, attn = gat(x, edge_index, edge_attr=edge_embeddings, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                x = gat(x, edge_index, edge_attr=edge_embeddings)

            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        # Edge-level prediction
        # For each edge, concatenate [src_node, dst_node, edge_features]
        src_nodes = x[edge_index[0]]  # Source node embeddings
        dst_nodes = x[edge_index[1]]  # Destination node embeddings

        edge_input = torch.cat([src_nodes, dst_nodes, edge_embeddings, edge_attr], dim=1)
        edge_logits = self.edge_classifier(edge_input)

        if return_attention:
            return edge_logits, attention_weights
        return edge_logits


class GNNFraudDetectorWrapper:
    """
    Wrapper class for training and inference with GNN
    Compatible with existing ensemble architecture
    """

    def __init__(self, node_features: int = 7, edge_features: int = 7, hidden_dim: int = 64):
        """Initialize GNN wrapper - FIXED: 7 node features (was 8, removed fraud rate)"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GNNFraudDetector(node_features, edge_features, hidden_dim).to(self.device)
        self.graph_builder = TemporalGraphBuilder()

    def fit(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 1, lr: float = 0.001):
        """
        Train GNN on transaction data

        Args:
            df: Transaction DataFrame
            epochs: Training epochs
            batch_size: Batch size (usually 1 for full graph)
            lr: Learning rate
        """
        # Build graph
        graph = self.graph_builder.build_graph_from_transactions(df)
        graph = graph.to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # Class weights for imbalanced data
        fraud_count = graph.y.sum().item()
        normal_count = len(graph.y) - fraud_count
        weights = torch.FloatTensor([1.0, normal_count / (fraud_count + 1)]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Training loop
        self.model.train()
        best_loss = float('inf')

        print(f"Training GNN on graph with {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            logits = self.model(graph.x, graph.edge_index, graph.edge_attr)
            loss = criterion(logits, graph.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                accuracy = (preds == graph.y).float().mean()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

            if loss.item() < best_loss:
                best_loss = loss.item()

        print(f"Training complete. Best loss: {best_loss:.4f}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability for transactions

        Args:
            df: Transaction DataFrame

        Returns:
            Fraud probabilities for each transaction
        """
        self.model.eval()

        # Build graph
        graph = self.graph_builder.build_graph_from_transactions(df)
        graph = graph.to(self.device)

        with torch.no_grad():
            logits = self.model(graph.x, graph.edge_index, graph.edge_attr)
            probs = F.softmax(logits, dim=1)[:, 1]  # Fraud probability

        return probs.cpu().numpy()

    def get_attention_explanations(self, df: pd.DataFrame) -> Dict:
        """
        Get attention-based explanations

        Returns:
            Attention weights showing which neighboring transactions influence prediction
        """
        self.model.eval()

        graph = self.graph_builder.build_graph_from_transactions(df)
        graph = graph.to(self.device)

        with torch.no_grad():
            logits, attention_weights = self.model(
                graph.x, graph.edge_index, graph.edge_attr, return_attention=True
            )

        return {
            'predictions': F.softmax(logits, dim=1).cpu().numpy(),
            'attention_weights': [attn.cpu().numpy() for attn in attention_weights],
            'fraud_patterns': self.graph_builder.detect_fraud_patterns(df)
        }

    def save(self, path: str):
        """Save GNN model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'graph_builder': self.graph_builder
        }, path)

    @staticmethod
    def load(path: str, node_features: int = 7, edge_features: int = 7, hidden_dim: int = 64):
        """Load GNN model - FIXED: 7 node features (was 8)"""
        wrapper = GNNFraudDetectorWrapper(node_features, edge_features, hidden_dim)
        checkpoint = torch.load(path, weights_only=False)
        wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        wrapper.graph_builder = checkpoint['graph_builder']
        return wrapper
