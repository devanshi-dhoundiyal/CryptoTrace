# CryptoTrace – AI-Based Cryptocurrency Fraud Detection and Investigation System

CryptoTrace is a machine learning-based cryptocurrency transaction classification and investigation system developed using the Elliptic Bitcoin dataset.

The project uses XGBoost to classify potentially illicit Bitcoin transactions and estimate their illicit probability. It also includes a temporal Graph Convolutional Network (GCN) for comparative research and a transaction network visualization interface.

## Features

- Cryptocurrency transaction classification using XGBoost.
- Illicit transaction probability estimation.
- Temporal GCN model for research comparison.
- Transaction ID search and investigation.
- Transaction network visualization.
- Dashboard showing dataset and prediction statistics.
- FastAPI backend and React frontend.

## Technology Stack

**Machine Learning:** Python, Pandas, XGBoost, PyTorch Geometric, NetworkX

**Backend:** FastAPI, Python

**Frontend:** React, TypeScript, Vite, Tailwind CSS, React Flow

**Dataset:** Elliptic Bitcoin Dataset

## Project Structure

```text
CryptoTrace/
├── backend/
├── frontend/
├── dataset/
├── README.md
└── .gitignore
```

## Dataset

The project uses the Elliptic Bitcoin dataset, containing transaction features, class labels, and recorded transaction relationships.

Dataset files:

- elliptic_txs_features.csv
- elliptic_txs_classes.csv
- elliptic_txs_edgelist.csv

## Model Performance

The XGBoost model was evaluated using a chronological train-test split.

| Metric | XGBoost |
|---|---:|
| Accuracy | 97.28% |
| Precision | 82.68% |
| Recall | 73.59% |
| F1-score | 0.7787 |
| ROC-AUC | 0.9292 |
| PR-AUC | 0.7994 |

A temporal GCN was also implemented as a comparative research model.

## Running the Project

Backend and frontend setup instructions will be added here.

## Disclaimer

CryptoTrace is an academic research project. Model predictions indicate potentially suspicious activity and should not be treated as definitive evidence of fraud or illegal activity.