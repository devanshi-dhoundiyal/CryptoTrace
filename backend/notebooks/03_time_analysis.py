import pandas as pd

# ==========================================
# LOAD PREPARED DATA
# ==========================================

data = pd.read_csv("results/labelled_transactions.csv")


# ==========================================
# BASIC TIME-STEP INFORMATION
# ==========================================

print("\n========== TIME STEP INFORMATION ==========")

print("Minimum time step:", data["time_step"].min())
print("Maximum time step:", data["time_step"].max())

print("\nNumber of unique time steps:")
print(data["time_step"].nunique())


# ==========================================
# TRANSACTIONS PER TIME STEP
# ==========================================

print("\n========== TRANSACTIONS PER TIME STEP ==========")

time_counts = data["time_step"].value_counts().sort_index()

print(time_counts)


# ==========================================
# CLASS DISTRIBUTION BY TIME STEP
# ==========================================

print("\n========== CLASS DISTRIBUTION BY TIME STEP ==========")

time_class_distribution = pd.crosstab(
    data["time_step"],
    data["target"]
)

print(time_class_distribution)


# ==========================================
# ILlicit PERCENTAGE BY TIME STEP
# ==========================================

print("\n========== ILLICIT PERCENTAGE BY TIME STEP ==========")

illicit_percentage = (
    data.groupby("time_step")["target"]
    .mean() * 100
)

print(illicit_percentage)