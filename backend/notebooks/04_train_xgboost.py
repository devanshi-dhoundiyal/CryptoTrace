
import pandas as pd
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# 1. Load prepared data
data = pd.read_csv("results/labelled_transactions.csv")

print("Dataset loaded:", data.shape)

# 2. Split chronologically
# Earlier time steps for training; later steps for testing
train_data = data[data["time_step"] <= 34].copy()
test_data = data[data["time_step"] >= 35].copy()

# 3. Select the 165 transaction features
feature_columns = [
    col for col in data.columns
    if col.startswith("feature_")
]

X_train = train_data[feature_columns]
y_train = train_data["target"]

X_test = test_data[feature_columns]
y_test = test_data["target"]

print("\nTRAINING DATA")
print("Rows:", len(train_data))
print("Time steps: 1–34")
print("Class counts:")
print(y_train.value_counts().sort_index())

print("\nTEST DATA")
print("Rows:", len(test_data))
print("Time steps: 35–49")
print("Class counts:")
print(y_test.value_counts().sort_index())

# 4. Account for class imbalance
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

if positive_count == 0 or negative_count == 0:
    raise ValueError("Training data must contain both classes.")

# 5. Create model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=negative_count / positive_count,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# 6. Train
print("\nTraining XGBoost...")
model.fit(X_train, y_train)
print("Training completed!")

# 7. Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 8. Evaluate
print("\n========== XGBOOST RESULTS ==========")
print(f"Accuracy  : {accuracy_score(y_test, predictions):.4f}")
print(f"Precision : {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1-score  : {f1_score(y_test, predictions, zero_division=0):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, probabilities):.4f}")
print(f"PR-AUC    : {average_precision_score(y_test, probabilities):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Licit", "Illicit"],
    zero_division=0
))

# 9. Save trained model
model.save_model("models/xgboost_crypto_fraud_model.json")
print("\nModel saved: models/xgboost_crypto_fraud_model.json")

# 10. Save predictions for later investigation
output = test_data[["txId", "time_step", "target"]].copy()
output["predicted_class"] = predictions
output["illicit_probability"] = probabilities

output.to_csv("results/xgboost_predictions.csv", index=False)
print("Predictions saved: results/xgboost_predictions.csv")
