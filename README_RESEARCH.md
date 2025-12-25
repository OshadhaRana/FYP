# Human-Centric Explainable AI for Financial Crime Detection

**Research Project**: Final Year Research
**Focus**: Explainable Graph Neural Networks with Attention-Based Interpretability
**Status**: 🚧 In Progress

---

## Project Structure

```
xai-fincrime-poc-starter/
│
├── archive/                          # Archived baseline ensemble system
│   └── baseline_ensemble_YYYYMMDD/   # Timestamped archive
│
├── research/                         # NEW: Research implementation
│   ├── explainable_gnn/              # Explainable GNN architecture
│   ├── explainability/               # XAI methods (attention, patterns)
│   ├── visualization/                # Graph visualization and dashboards
│   ├── evaluation/                   # Evaluation metrics and user study
│   └── baselines/                    # Baseline XAI methods (SHAP, LIME)
│
├── experiments/                      # NEW: Experimental code
│   ├── attention_analysis/           # Attention mechanism experiments
│   ├── pattern_detection/            # Fraud pattern detection
│   ├── case_studies/                 # Detailed case studies
│   ├── comparison/                   # GNN vs traditional XAI
│   └── user_study/                   # User study materials
│
├── docs/                             # NEW: Research documentation
│   ├── literature_review/            # Related work
│   ├── methodology/                  # Research methods
│   ├── results/                      # Experimental results
│   ├── figures/                      # Paper figures
│   └── paper/                        # Research paper drafts
│
├── notebooks/                        # NEW: Jupyter notebooks
│   ├── exploratory/                  # Exploratory analysis
│   ├── visualization/                # Visualization experiments
│   └── analysis/                     # Results analysis
│
├── backend/                          # Original backend (preserved)
│   └── app/
│       ├── models/                   # Original models (baseline)
│       └── utils/                    # Utilities
│
├── data/
│   ├── processed/                    # Processed datasets
│   ├── models/                       # Trained models
│   ├── explanations/                 # NEW: Generated explanations
│   └── visualizations/               # NEW: Generated visualizations
│
└── utils/                            # Shared utilities

```

---

## Research Objective

Develop an **Explainable Graph Neural Network (GNN)** that provides human-interpretable
explanations for financial fraud detection, enabling fraud investigators to understand
**WHY** transactions are flagged as fraudulent.

---

## Novel Contributions

1. **Attention-Based Explainable GNN**: Extract and visualize attention weights
2. **Fraud Pattern Detection**: Detect circular flows, fan-out patterns, rapid sequences
3. **Human-Centric Visualization**: Interactive graph interface for investigators
4. **User Evaluation**: Validate explanations with domain experts

---

## Baseline System (Archived)

The baseline ensemble system (99.96% ROC-AUC) is archived in:
- `archive/baseline_ensemble_YYYYMMDD/`

This serves as a comparison baseline for the Explainable GNN research.

---

## Quick Start

### 1. Set Up Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Baseline Comparison (Optional)

```bash
# Load archived baseline models
python experiments/comparison/load_baseline.py
```

### 3. Train Explainable GNN

```bash
# Train GNN with attention-based explainability
python research/explainable_gnn/train.py
```

### 4. Generate Explanations

```bash
# Generate explanations for test transactions
python research/explainability/generate_explanations.py
```

### 5. Visualize Results

```bash
# Launch visualization dashboard
streamlit run research/visualization/dashboard.py
```

---

## Research Questions

1. **RQ1**: How do attention mechanisms in GNNs identify suspicious transaction patterns?
2. **RQ2**: What network structures (fraud rings, money laundering) are most indicative of fraud?
3. **RQ3**: How can graph-based explanations be visualized for fraud investigators?
4. **RQ4**: Does explainability improve fraud investigator trust and decision-making?

---

## Methodology

### Phase 1: Explainable GNN Development (Weeks 1-4)
- Extend GNN with attention-based explainability
- Implement fraud pattern detection
- Add natural language explanation generation

### Phase 2: Visualization Interface (Weeks 5-6)
- Design interactive graph visualization
- Create attention heatmaps
- Build prototype dashboard

### Phase 3: Comparative Analysis (Weeks 7-8)
- Compare GNN vs SHAP/LIME explanations
- Analyze fraud pattern detection
- Case studies

### Phase 4: User Evaluation (Weeks 9-10)
- Design user study
- Conduct evaluation with domain experts
- Analyze results

### Phase 5: Writing (Weeks 11-12)
- Write research paper/thesis
- Create presentation
- Document findings

---

## Key Files

### Research Implementation
- `research/explainable_gnn/model.py` - Explainable GNN architecture
- `research/explainability/attention_extractor.py` - Attention extraction
- `research/explainability/pattern_detector.py` - Fraud pattern detection
- `research/visualization/graph_viz.py` - Graph visualization
- `research/visualization/dashboard.py` - Streamlit dashboard

### Experiments
- `experiments/attention_analysis/` - Attention mechanism analysis
- `experiments/pattern_detection/` - Pattern detection experiments
- `experiments/case_studies/` - Detailed fraud case analysis
- `experiments/comparison/gnn_vs_shap.py` - Comparison experiment

### Documentation
- `docs/RESEARCH_PROPOSAL.md` - Research proposal
- `docs/literature_review/` - Related work summary
- `docs/methodology/` - Research methods
- `docs/results/` - Experimental results

---

## Performance Targets

### Baseline (Archived)
- Ensemble: 99.96% ROC-AUC
- No explainability

### Research Goal
- Explainable GNN: ~94% ROC-AUC (acceptable tradeoff)
- **With human-interpretable explanations**

---

## Evaluation Metrics

### Model Performance
- ROC-AUC, Precision, Recall, F1-Score

### Explainability Quality
- **Fidelity**: How well explanations reflect model behavior
- **Consistency**: Similar transactions → similar explanations
- **Sparsity**: Concise explanations
- **Comprehensibility**: User study ratings
- **Trust**: User study ratings
- **Actionability**: User study ratings

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Literature Review | Related work summary |
| 3-4 | Explainable GNN | Working implementation |
| 5-6 | Visualization | Interactive dashboard |
| 7-8 | Comparison | GNN vs SHAP analysis |
| 9-10 | User Study | Evaluation results |
| 11-12 | Writing | Research paper/thesis |

---

## Related Work

### Graph Neural Networks for Fraud
- GeniePath (KDD 2019)
- Heterogeneous GNN for Fraud (AAAI 2020)
- CARE-GNN (CIKM 2021)

### Explainable AI for Fraud
- GNNExplainer (NeurIPS 2019)
- XAI for Financial Crime (IEEE 2020)

### Human-Centric XAI
- Human-Centered XAI (CHI 2020)
- Explanation in AI (AI Journal 2019)

---

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- Streamlit (dashboard)
- NetworkX (graph visualization)
- Plotly (interactive plots)
- SHAP/LIME (baseline comparison)

---

## Contact

For questions or collaboration:
- Research Proposal: `docs/RESEARCH_PROPOSAL.md`
- Archived Baseline: `archive/baseline_ensemble_YYYYMMDD/README.md`

---

**Research Status**: 🚧 Active Development
**Last Updated**: {datetime.now().strftime("%Y-%m-%d")}
