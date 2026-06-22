# GNN-Based Bitcoin Fraud Detection
### Elliptic Dataset — End-to-End Pipeline  

This project leverages **Graph Neural Networks (GNNs)** to detect illicit Bitcoin transactions using the **Elliptic dataset**. It implements a complete pipeline covering data preprocessing, graph construction, model training, and evaluation.

---

## Dataset  

Download the dataset from Kaggle:  
[Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Place the following files inside a `data/` directory:

- elliptic_txs_features.csv  
- elliptic_txs_edgelist.csv  
- elliptic_txs_classes.csv  

---

## Pipeline Overview  

### 1. Data Preprocessing  
- Merged transaction features with labels  
- Label mapping:  
  - 1 → illicit  
  - 2 → licit  
  - unknown → -1 (excluded from training)  
- Normalised 165 transaction features  
- Created masks for labelled nodes  

---

### 2. Graph Construction  
- Nodes → transactions  
- Edges → transaction flow  
- Built using PyTorch Geometric  

Key components:  
- x → node features  
- edge_index → graph connectivity  
- y → labels  
- train/test masks  

---

### 3. Model  
- Graph Neural Networks (GNNs)  
- Learns node representations via neighbourhood aggregation  

Captures:  
- Local features (transaction attributes)  
- Global structure (network connectivity)  

---

### 4. Experiments  
- Full feature model (165 features)  
- Implemented:
  - GCN  
  - GraphSAGE  
  - GAT  

---

## Results  

The model demonstrates:  

- Effective detection of illicit transactions  
- Improved performance using graph structure over standalone features  
- Ability to capture hidden relationships in transaction networks  

### Observations  
- GNNs outperform traditional ML methods  
- Graph structure significantly improves classification  
- Class imbalance affects performance  
- Connectivity patterns are crucial for fraud detection  

---

## Visualisations  

- Class distribution (pie chart)  
- Feature correlation heatmap  
- Graph degree distribution  

---

## Key Insights  

- Fraud detection is naturally a graph-based problem  
- Transactions are interdependent  
- Structural information improves predictive performance  

---

## Notes  

- Dataset is highly imbalanced  
- Unknown labels are excluded during training  

``
