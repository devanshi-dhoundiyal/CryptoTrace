
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Load the trained model
model = XGBClassifier()
model.load_model("models/xgboost_crypto_fraud_model.json")

# Extract feature importance
feature_names = [
    f"feature_{i}" for i in range(1, 166)
]

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

# Sort from most important to least important
importance = importance.sort_values(
    "importance", ascending=False
)

print("\n========== TOP 20 IMPORTANT FEATURES ==========")
print(importance.head(20).to_string(index=False))

# Save all feature importance values
importance.to_csv(
    "results/xgboost_feature_importance.csv",
    index=False
)

# Plot the top 20 features
top20 = importance.head(20).sort_values("importance")

plt.figure(figsize=(10, 7))
plt.barh(top20["feature"], top20["importance"])
plt.xlabel("XGBoost Feature Importance")
plt.ylabel("Transaction Feature")
plt.title("Top 20 Features - Cryptocurrency Fraud Detection")
plt.tight_layout()

plt.savefig(
    "results/xgboost_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance chart saved successfully.")

