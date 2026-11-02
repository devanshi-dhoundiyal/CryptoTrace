
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Paths and reproducibility
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT.parent / "elliptic_bitcoin_dataset"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

FEATURES_PATH = DATASET / "elliptic_txs_features.csv"
CLASSES_PATH = DATASET / "elliptic_txs_classes.csv"
EDGES_PATH = DATASET / "elliptic_txs_edgelist.csv"

MODEL_PATH = MODELS / "gcn_crypto_fraud_temporal.pt"
PREDICTIONS_PATH = RESULTS / "gcn_temporal_predictions.csv"

SEED = 42
CUTOFF = 34
EPOCHS = 50

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# 2. Load the Elliptic dataset
# --------------------------------------------------

print("\nLoading dataset...")

features_df = pd.read_csv(FEATURES_PATH, header=None)
features_df.columns = (
    ["txId", "time_step"]
    + [f"feature_{i}" for i in range(1, 166)]
)

classes_df = pd.read_csv(CLASSES_PATH)
edges_df = pd.read_csv(EDGES_PATH)

# Normalize transaction IDs and labels.
features_df["txId"] = features_df["txId"].astype("int64")
classes_df["txId"] = classes_df["txId"].astype("int64")

classes_df["class"] = (
    classes_df["class"].astype(str).str.strip().str.lower()
)

# Elliptic labels: 1 = illicit, 2 = licit.
label_map = {"1": 1, "2": 0}
classes_df["label"] = classes_df["class"].map(label_map)
label_lookup = classes_df.set_index("txId")["label"]

features_df["label"] = features_df["txId"].map(label_lookup)

feature_cols = [f"feature_{i}" for i in range(1, 166)]

X = features_df[feature_cols].to_numpy(dtype=np.float32)
time_steps = features_df["time_step"].to_numpy(dtype=np.int64)
tx_ids = features_df["txId"].to_numpy(dtype=np.int64)

# Unknown transactions have label NaN.
y = (
    features_df["label"]
    .fillna(-1)
    .to_numpy(dtype=np.int64)
)

# Map transaction IDs to their row indices.
tx_to_idx = {
    tx_id: idx for idx, tx_id in enumerate(tx_ids)
}

# Read the directed transaction edge list.
src_ids = edges_df["txId1"].astype("int64").to_numpy()
dst_ids = edges_df["txId2"].astype("int64").to_numpy()

src_idx = pd.Series(src_ids).map(tx_to_idx).to_numpy()
dst_idx = pd.Series(dst_ids).map(tx_to_idx).to_numpy()

# Discard edges referencing transactions absent from features.
valid_edges = ~pd.isna(src_idx) & ~pd.isna(dst_idx)
src_idx = src_idx[valid_edges].astype(np.int64)
dst_idx = dst_idx[valid_edges].astype(np.int64)

# Add reverse edges for undirected message passing.
all_src = np.concatenate([src_idx, dst_idx])
all_dst = np.concatenate([dst_idx, src_idx])

X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.long, device=device)
time_tensor = torch.tensor(time_steps, dtype=torch.long, device=device)

print(f"Transactions: {len(tx_ids):,}")
print(f"Features per transaction: {X.shape[1]}")
print(f"Valid original edges: {len(src_idx):,}")
print(f"Training cutoff: time step {CUTOFF}")

# --------------------------------------------------
# 3. Build a graph snapshot for a given time step
# --------------------------------------------------

def build_snapshot(max_time):
    """
    Include only nodes and edges available by max_time.
    Edges are retained only when both endpoint transactions
    have time steps <= max_time.
    """
    active_nodes = np.flatnonzero(time_steps <= max_time)

    edge_mask = (
        (time_steps[all_src] <= max_time)
        & (time_steps[all_dst] <= max_time)
    )

    edge_index = torch.tensor(
        np.vstack([all_src[edge_mask], all_dst[edge_mask]]),
        dtype=torch.long,
        device=device,
    )

    return active_nodes, edge_index


# Training snapshot: no transactions after step 34.
train_nodes, train_edge_index = build_snapshot(CUTOFF)

train_node_mask = y[train_nodes] >= 0
train_nodes_labelled = train_nodes[train_node_mask]

train_global_to_local = np.full(len(tx_ids), -1, dtype=np.int64)
train_global_to_local[train_nodes] = np.arange(len(train_nodes))

train_edge_index_local = torch.tensor(
    train_global_to_local[train_edge_index.cpu().numpy()],
    dtype=torch.long,
    device=device,
)

train_features = X_tensor[train_nodes]
train_labels = y_tensor[train_nodes]

train_label_mask = train_labels >= 0

print(f"Training snapshot nodes: {len(train_nodes):,}")
print(f"Labelled training transactions: {int(train_label_mask.sum()):,}")
print(f"Training snapshot edges: {train_edge_index.shape[1]:,}")

# --------------------------------------------------
# 4. Define the GCN model
# --------------------------------------------------

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return self.output(x)


model = GCN(input_dim=X.shape[1]).to(device)

# Weight the minority illicit class more heavily.
labelled_train_y = y[train_nodes][train_node_mask]
class_counts = np.bincount(labelled_train_y, minlength=2)

class_weights = torch.tensor(
    [
        len(labelled_train_y) / (2 * max(class_counts[0], 1)),
        len(labelled_train_y) / (2 * max(class_counts[1], 1)),
    ],
    dtype=torch.float32,
    device=device,
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)

# --------------------------------------------------
# 5. Train using only the historical snapshot
# --------------------------------------------------

print("\nTraining GCN on time steps 1-34...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(train_features, train_edge_index_local)

    loss = criterion(
        logits[train_label_mask],
        train_labels[train_label_mask],
    )

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:02d}/{EPOCHS}, loss: {loss.item():.4f}")

# Save a separate model so the original model remains untouched.
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nSaved temporal GCN model: {MODEL_PATH}")

# --------------------------------------------------
# 6. Evaluate each future time step separately
# --------------------------------------------------

print("\nEvaluating time steps 35-49...")

model.eval()
prediction_rows = []

with torch.no_grad():
    for current_time in range(CUTOFF + 1, int(time_steps.max()) + 1):

        # This snapshot includes transactions only up to current_time.
        active_nodes, snapshot_edge_index = build_snapshot(current_time)

        global_to_local = np.full(len(tx_ids), -1, dtype=np.int64)
        global_to_local[active_nodes] = np.arange(len(active_nodes))

        snapshot_edge_index_local = torch.tensor(
            global_to_local[snapshot_edge_index.cpu().numpy()],
            dtype=torch.long,
            device=device,
        )

        snapshot_features = X_tensor[active_nodes]

        logits = model(
            snapshot_features,
            snapshot_edge_index_local,
        )

        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        # Score only labelled transactions from the current time step.
        current_mask = (
            (time_steps[active_nodes] == current_time)
            & (y[active_nodes] >= 0)
        )

        current_local_idx = np.flatnonzero(current_mask)
        current_global_idx = active_nodes[current_local_idx]

        current_probabilities = probabilities[current_local_idx]
        current_predictions = (
            current_probabilities >= 0.5
        ).astype(np.int64)

        for i, global_idx in enumerate(current_global_idx):
            prediction_rows.append({
                "txId": int(tx_ids[global_idx]),
                "time_step": int(time_steps[global_idx]),
                "actual": int(y[global_idx]),
                "illicit_probability": float(current_probabilities[i]),
                "predicted": int(current_predictions[i]),
            })

        print(
            f"Time step {current_time}: "
            f"{len(current_global_idx):,} labelled transactions evaluated"
        )

# --------------------------------------------------
# 7. Save predictions and calculate metrics
# --------------------------------------------------

predictions_df = pd.DataFrame(prediction_rows)
predictions_df.to_csv(PREDICTIONS_PATH, index=False)

if predictions_df.empty:
    raise RuntimeError("No labelled test transactions were evaluated.")

y_true = predictions_df["actual"].to_numpy()
y_pred = predictions_df["predicted"].to_numpy()
y_prob = predictions_df["illicit_probability"].to_numpy()

print("\n========== TEMPORAL GCN RESULTS ==========")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(
    f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}"
)
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_true, y_prob):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Licit", "Illicit"],
        zero_division=0,
    )
)

print(f"\nPredictions saved to: {PREDICTIONS_PATH}")
print("\nTemporal GCN evaluation completed.")
