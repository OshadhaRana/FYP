# MVP Test Cases for Screen Recording Demonstration

## Dashboard URL
**Open in browser**: http://localhost:8501

---

## Test Case 1: High-Confidence Fraud Detection (TRANSFER)

### Transaction Details
Navigate to **Transaction Analysis** and enter these values or find a similar transaction:

| Field | Value |
|-------|-------|
| **Transaction Index** | Find index with `isFraud = 1` and `type = TRANSFER` |
| **Type** | TRANSFER |
| **Amount** | ~$500,000 - $1,000,000 |
| **Expected Result** | FRAUD (>90% probability) |

### What to Show in Demo
1. Select "Fraud Only" filter
2. Click "Random Transaction" to get a fraud case
3. Show the **Transaction Details** table
4. Highlight the **FRAUD** badge with high probability
5. Show the **Model Comparison** bar chart (XGBoost vs RF vs GNN)
6. Explain the **SHAP Waterfall** plot - which features contributed
7. Read the **Natural Language Explanation** aloud

### Sample Fraud Transaction (Index 85003)
```
Type: TRANSFER
Amount: $566,142.89
Origin Balance: $566,142.89 (100% utilization)
Destination Balance: $0 (empty account)
Result: FRAUD 99.9%
```

### Key Points to Mention
- "Transaction type TRANSFER has 36% fraud rate"
- "Amount is 3x larger than normal average"
- "Account being drained 100% - classic fraud indicator"
- "All 3 models agree this is fraud"

---

## Test Case 2: Large CASH_OUT Fraud

### Transaction Details
| Field | Value |
|-------|-------|
| **Type** | CASH_OUT |
| **Amount** | >$1,000,000 |
| **Balance Utilization** | 100% |
| **Expected Result** | FRAUD (100% probability) |

### What to Show
1. Find a high-value CASH_OUT fraud
2. Show the extreme amount in the explanation
3. Highlight that XGBoost gives 100% confidence
4. Explain GNN might give slightly lower (94%) due to network context

### Key Points to Mention
- "CASH_OUT with complete account drain is highly suspicious"
- "Amount $1.9M is well above the $1.48M fraud average"
- "This matches known fraud patterns in the dataset"

---

## Test Case 3: Normal Transaction (Low Risk)

### Transaction Details
| Field | Value |
|-------|-------|
| **Filter** | Normal Only |
| **Type** | PAYMENT |
| **Amount** | <$10,000 |
| **Expected Result** | NORMAL (<5% probability) |

### What to Show
1. Select "Normal Only" filter
2. Find a PAYMENT transaction
3. Show the **NORMAL** badge with low probability
4. Explain why it's classified as normal:
   - PAYMENT type has 0% fraud rate
   - Small amount
   - Normal balance patterns

### Key Points to Mention
- "PAYMENT transactions have 0% fraud rate in our dataset"
- "Small amount within normal range"
- "The system correctly identifies low-risk transactions"

---

## Test Case 4: GNN Network Visualization

### Navigation
Go to **Graph Visualization** page

### What to Show
1. Explain the graph represents transaction network
2. **Red edges** = Fraudulent transactions
3. **Red nodes** = Accounts involved in fraud
4. Enable "Highlight Fraud" checkbox
5. Zoom in on a fraud cluster

### Key Points to Mention
- "GNN sees relationships between accounts that XGBoost cannot"
- "This graph shows potential fraud rings"
- "Connected accounts may indicate coordinated fraud"
- "This is the novel research contribution"

---

## Test Case 5: Model Comparison View

### Navigation
Go to **Model Comparison** page

### What to Show
1. The 3-model architecture table:
   - XGBoost (50%) - 99.99% AUC - Baseline
   - Random Forest (15%) - 99.98% AUC - Validation
   - GNN (35%) - 94.36% AUC - Research Focus

2. The **5% Accuracy Trade-off** section
3. The **Ensemble Pie Chart**

### Key Points to Mention
- "We simplified from 5 models to 3 for clearer research"
- "XGBoost is our baseline with SHAP explainability"
- "GNN is our research focus with graph-based explanations"
- "The 5% accuracy drop is justified by unique network insights"

---

## Test Case 6: Dashboard Overview

### Navigation
Go to **Dashboard Overview** page

### What to Show
1. Dataset statistics:
   - 100,000 transactions
   - 8,213 fraud cases (8.21%)
   - 3 active models

2. **Fraud by Transaction Type** chart
   - TRANSFER: 36% fraud rate
   - CASH_OUT: 12% fraud rate
   - PAYMENT: 0% fraud rate

3. **Model Performance Comparison** chart

### Key Points to Mention
- "Our dataset has 8% fraud rate - realistic imbalance"
- "Only TRANSFER and CASH_OUT have fraud"
- "All 3 models achieve >94% accuracy"

---

## Demo Script (5-7 minutes)

### Introduction (30 seconds)
"This is our Human-Centric Explainable AI system for Financial Crime Detection. The key innovation is combining traditional machine learning with Graph Neural Networks for dual explainability."

### Dashboard Overview (1 minute)
Show statistics and model architecture

### Fraud Detection Demo (2 minutes)
1. Test Case 1: Show high-confidence TRANSFER fraud
2. Explain SHAP waterfall plot
3. Read natural language explanation

### Normal Transaction (1 minute)
1. Test Case 3: Show a PAYMENT transaction
2. Explain why it's correctly classified as normal

### GNN Visualization (1 minute)
1. Show transaction graph
2. Highlight fraud clusters
3. Explain research contribution

### Model Comparison (1 minute)
1. Show 3-model architecture
2. Explain the 5% accuracy trade-off
3. Summarize research question

### Conclusion (30 seconds)
"This system demonstrates that graph-based explainability provides unique value that traditional ML cannot offer, justifying a small accuracy trade-off for better human-centric explanations."

---

## Quick Reference: Transaction Indices

### Known Fraud Cases (for demo)
Run this to find good demo cases:
```python
import pandas as pd
df = pd.read_csv('data/processed/paysim_sample.csv')

# High-value TRANSFER fraud
transfer_fraud = df[(df['isFraud']==1) & (df['type']=='TRANSFER')].nlargest(5, 'amount')
print("TRANSFER Fraud Cases:")
print(transfer_fraud[['type', 'amount', 'oldbalanceOrg']].to_string())

# High-value CASH_OUT fraud
cashout_fraud = df[(df['isFraud']==1) & (df['type']=='CASH_OUT')].nlargest(5, 'amount')
print("\nCASH_OUT Fraud Cases:")
print(cashout_fraud[['type', 'amount', 'oldbalanceOrg']].to_string())
```

### Normal Cases (for comparison)
- Any PAYMENT transaction
- Any CASH_IN transaction
- Small amount (<$1000) transactions

---

## Troubleshooting

### If Dashboard Doesn't Load
```bash
cd c:\xai-fincrime-poc-starter
pip install streamlit plotly networkx
streamlit run dashboard/app.py
```

### If Models Don't Load
Check that these files exist:
- `data/models/xgb_model.joblib`
- `data/models/rf_model.joblib`
- `data/models/preprocessor.joblib`

---

## Key Messages for Submission

1. **Research Question**: "Can Graph Neural Networks provide better explainability than traditional ML?"

2. **Architecture**: Simplified 3-model ensemble (XGBoost 50%, RF 15%, GNN 35%)

3. **Trade-off**: 5% accuracy loss justified by unique graph-based explanations

4. **Human-Centric**: Explanations designed for fraud investigators, not data scientists

5. **Dual Explainability**:
   - SHAP = "Which features matter?"
   - GNN = "Which accounts and relationships matter?"
