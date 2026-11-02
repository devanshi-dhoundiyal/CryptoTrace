import pandas as pd

# ==========================================
# DATASET PATHS
# ==========================================

features_path = "../elliptic_bitcoin_dataset/elliptic_txs_features.csv"
classes_path = "../elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
edges_path = "../elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"


# ==========================================
# LOAD FEATURES
# ==========================================

# The features CSV does NOT have a header row.
features = pd.read_csv(
    features_path,
    header=None
)

# Create column names
feature_columns = ["txId", "time_step"]

for i in range(1, 166):
    feature_columns.append(f"feature_{i}")

features.columns = feature_columns


# ==========================================
# LOAD CLASSES
# ==========================================

classes = pd.read_csv(classes_path)


# ==========================================
# LOAD EDGES
# ==========================================

edges = pd.read_csv(edges_path)


# ==========================================
# DISPLAY FEATURES
# ==========================================

print("\n========== FEATURES ==========")

print("Shape:", features.shape)

print("\nColumns:")
print(features.columns.tolist())

print("\nFirst 5 rows:")
print(features.head())


# ==========================================
# DISPLAY CLASSES
# ==========================================

print("\n========== CLASSES ==========")

print("Shape:", classes.shape)

print("\nColumns:")
print(classes.columns.tolist())

print("\nFirst 5 rows:")
print(classes.head())

print("\nClass distribution:")
print(classes["class"].value_counts())


# ==========================================
# DISPLAY EDGES
# ==========================================

print("\n========== EDGES ==========")

print("Shape:", edges.shape)

print("\nColumns:")
print(edges.columns.tolist())

print("\nFirst 5 rows:")
print(edges.head())


# ==========================================
# BASIC CHECKS
# ==========================================

print("\n========== BASIC CHECKS ==========")

print("Missing values in features:",
      features.isnull().sum().sum())

print("Missing values in classes:",
      classes.isnull().sum().sum())

print("Missing values in edges:",
      edges.isnull().sum().sum())

print("Duplicate transaction IDs in features:",
      features["txId"].duplicated().sum())

print("Duplicate transaction IDs in classes:",
      classes["txId"].duplicated().sum())