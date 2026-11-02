
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
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

# ==========================================
# 1. SETTINGS AND PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT.parent / "elliptic_bitcoin_dataset"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")
print("Device:", device)


# ==========================================
# 2. LOAD DATA
# ==========================================

features_path = DATASET_DIR / "elliptic_txs_features.csv"
classes_path = DATASET_DIR / "elliptic_txs_classes.csv"
edges_path = DATASET_DIR / "elliptic_txs_edgelist.csv"

features_df = pd.read_csv(features_path, header=None)
classes_df = pd.read_csv(classes_path)
edges_df = pd.read_csv(edges_path)

# The original feature CSV has no header.
# Column 0 = txId, column 1 = time_step,
# columns 2 onward = 165 features.
features_df.columns = (
    ["txId", "time_step"]
    + [f"feature_{i}" for i in range(1, 166)]
)

print("\n========== DATA SHAPES ==========")
print("Features:", features_df.shape)
print("Classes:", classes_df.shape)
print("Edges:", edges_df.shape)


# ==========================================
# 3. MAP CLASS LABELS
# ==========================================

# Original Elliptic class labels:
# "1" = illicit, "2" = licit, "unknown" = unknown
# Target: 0 = licit, 1 = illicit, -1 = unknown

classes_df["class"] = classes_df["class"].astype(str)

class_mapping = {
    "1": 1,
    "2": 0,
    "unknown": -1,
}

classes_df["target"] = classes_df["class"].map(class_mapping)

if classes_df["target"].isna().any():
    raise ValueError("Unexpected class labels found.")

data_df = features_df.merge(
    classes_df[["txId", "target"]],
    on="txId",
    how="left",
    validate="one_to_one",
)

if data_df["target"].isna().any():
    raise ValueError("Some transactions do not have class entries.")

data_df["target"] = data_df["target"].astype(int)

print("\nTarget distribution:")
print(data_df["target"].value_counts())


# ==========================================
# 4. MAP TRANSACTION IDS TO NODE INDICES
# ==========================================

# Every transaction gets a node index from 0 to N-1.
tx_ids = data_df["txId"].to_numpy()
tx_to_idx = {
    int(tx_id): idx
    for idx, tx_id in enumerate(tx_ids)
}

edge_src = edges_df["txId1"].map(tx_to_idx)
edge_dst = edges_df["txId2"].map(tx_to_idx)

valid_edges = edge_src.notna() & edge_dst.notna()

edge_src = edge_src[valid_edges].to_numpy(dtype=np.int64)
edge_dst = edge_dst[valid_edges].to_numpy(dtype=np.int64)

edge_index = torch.tensor(
    np.vstack([edge_src, edge_dst]),
    dtype=torch.long,
)

# GCNConv treats the graph as undirected by default only
# if edges are supplied in both directions, so add reverses.
edge_index = torch.cat(
    [edge_index, edge_index.flip(0)],
    dim=1,
)

print("\n========== GRAPH ==========")
print("Number of nodes:", len(data_df))
print("Directed edges loaded:", len(edge_src))
print("Edges supplied to GCN:", edge_index.shape[1])


# ==========================================
# 5. CREATE FEATURES AND LABELS
# ==========================================

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

X = data_df[feature_columns].to_numpy(dtype=np.float32)
y = data_df["target"].to_numpy(dtype=np.int64)
time_steps = data_df["time_step"].to_numpy()

# Defensive check for invalid numeric values
if not np.isfinite(X).all():
    raise ValueError("Features contain NaN or infinite values.")

x_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
time_tensor = torch.tensor(time_steps, dtype=torch.long)

# Only labelled transactions can be used in supervised loss.
train_mask = (time_tensor <= 34) & (y_tensor >= 0)
test_mask = (time_tensor >= 35) & (y_tensor >= 0)

print("\n========== SPLIT ==========")
print("Training labelled nodes:", int(train_mask.sum()))
print("Testing labelled nodes:", int(test_mask.sum()))

if not train_mask.any() or not test_mask.any():
    raise ValueError("Training or testing mask is empty.")


# ==========================================
# 6. BUILD GRAPH DATA OBJECT
# ==========================================

graph_data = Data(
    x=x_tensor,
    edge_index=edge_index,
    y=y_tensor,
)

graph_data = graph_data.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)


# ==========================================
# 7. DEFINE GCN MODEL
# ==========================================

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return self.output(x)


model = GCN(
    input_dim=x_tensor.shape[1],
    hidden_dim=64,
    dropout=0.3,
).to(device)

# Give the minority illicit class greater loss weight.
train_labels = y_tensor[train_mask]
class_counts = torch.bincount(train_labels, minlength=2)

if (class_counts == 0).any():
    raise ValueError("Training data must contain both classes.")

class_weights = (
    class_counts.sum().float()
    / (2.0 * class_counts.float())
).to(device)

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)


# ==========================================
# 8. TRAIN GCN
# ==========================================

print("\n========== TRAINING GCN ==========")

epochs = 50

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(
        graph_data.x,
        graph_data.edge_index,
    )

    loss = criterion(
        logits[train_mask],
        graph_data.y[train_mask],
    )

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"| Loss: {loss.item():.4f}"
        )

print("GCN training completed!")


# ==========================================
# 9. EVALUATE ON LATER TIME STEPS
# ==========================================

model.eval()

with torch.no_grad():
    logits = model(
        graph_data.x,
        graph_data.edge_index,
    )

    probabilities = torch.softmax(logits, dim=1)[:, 1]
    predictions = logits.argmax(dim=1)

y_test = graph_data.y[test_mask].cpu().numpy()
y_pred = predictions[test_mask].cpu().numpy()
y_probability = probabilities[test_mask].cpu().numpy()

print("\n========== GCN RESULTS ==========")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(
    f"Precision : "
    f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
)
print(
    f"Recall    : "
    f"{recall_score(y_test, y_pred, zero_division=0):.4f}"
)
print(f"F1-score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_probability):.4f}")
print(
    f"PR-AUC    : "
    f"{average_precision_score(y_test, y_probability):.4f}"
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Licit", "Illicit"],
        zero_division=0,
    )
)


# ==========================================
# 10. SAVE MODEL
# ==========================================

model_path = MODELS_DIR / "gcn_crypto_fraud_model.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": x_tensor.shape[1],
        "hidden_dim": 64,
        "dropout": 0.3,
        "class_mapping": {
            "0": "licit",
            "1": "illicit",
        },
    },
    model_path,
)

print("\nGCN model saved to:", model_path)


# ==========================================
# 11. SAVE TEST PREDICTIONS
# ==========================================

test_indices = torch.where(test_mask)[0].cpu().numpy()

output = pd.DataFrame({
    "txId": tx_ids[test_indices],
    "time_step": time_steps[test_indices],
    "actual_target": y_test,
    "predicted_class": y_pred,
    "illicit_probability": y_probability,
})

predictions_path = RESULTS_DIR / "gcn_predictions.csv"
output.to_csv(predictions_path, index=False)

print("GCN predictions saved to:", predictions_path)
print("\nGCN experiment completed.")
