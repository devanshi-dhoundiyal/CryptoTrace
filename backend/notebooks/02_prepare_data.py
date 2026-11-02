import pandas as pd

# ==========================================
# DATASET PATHS
# ==========================================

features_path = "../elliptic_bitcoin_dataset/elliptic_txs_features.csv"
classes_path = "../elliptic_bitcoin_dataset/elliptic_txs_classes.csv"


# ==========================================
# LOAD FEATURES
# ==========================================

features = pd.read_csv(
    features_path,
    header=None
)

# Create meaningful column names
feature_columns = ["txId", "time_step"]

for i in range(1, 166):
    feature_columns.append(f"feature_{i}")

features.columns = feature_columns


# ==========================================
# LOAD CLASSES
# ==========================================

classes = pd.read_csv(classes_path)


# ==========================================
# MERGE FEATURES + CLASSES
# ==========================================

data = features.merge(
    classes,
    on="txId",
    how="inner"
)


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("\n========== MERGED DATA ==========")

print("Shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())


# ==========================================
# CLASS DISTRIBUTION
# ==========================================

print("\n========== CLASS DISTRIBUTION ==========")

print(data["class"].value_counts())


# ==========================================
# REMOVE UNKNOWN LABELS
# ==========================================

labelled_data = data[
    data["class"].isin(["1", "2", 1, 2])
].copy()


print("\n========== LABELLED DATA ==========")

print("Shape:", labelled_data.shape)

print("\nClass distribution:")
print(labelled_data["class"].value_counts())


# ==========================================
# CONVERT LABELS
# ==========================================

# 1 = illicit
# 2 = licit

labelled_data["target"] = labelled_data["class"].map({
    1: 1,
    2: 0,
    "1": 1,
    "2": 0
})


print("\n========== TARGET DISTRIBUTION ==========")

print(labelled_data["target"].value_counts())


# ==========================================
# CHECK FOR MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

print(
    "Missing values:",
    labelled_data.isnull().sum().sum()
)


# ==========================================
# SAVE PREPARED DATA
# ==========================================

output_path = "results/labelled_transactions.csv"

labelled_data.to_csv(
    output_path,
    index=False
)

print("\nPrepared dataset saved to:")
print(output_path)