
import pandas as pd
import networkx as nx
from pathlib import Path

# ==========================================
# 1. DEFINE PROJECT PATHS
# ==========================================

# Project root: crypto_fraud_detection/
ROOT = Path(__file__).resolve().parent.parent

# Original Elliptic dataset is OUTSIDE the project folder,
# alongside crypto_fraud_detection inside research_project/
edges_path = (
    ROOT.parent
    / "elliptic_bitcoin_dataset"
    / "elliptic_txs_edgelist.csv"
)

# Prepared labelled data is inside crypto_fraud_detection/results/
data_path = ROOT / "results" / "labelled_transactions.csv"

# Output summary path
output_path = ROOT / "results" / "transaction_graph_summary.csv"


# ==========================================
# 2. CHECK FILE PATHS
# ==========================================

print("========== CHECKING FILE PATHS ==========")
print("Dataset edge file:", edges_path)
print("Prepared data file:", data_path)

if not edges_path.exists():
    raise FileNotFoundError(
        f"Edge file not found: {edges_path}\n"
        "Check the dataset folder name and CSV filename."
    )

if not data_path.exists():
    raise FileNotFoundError(
        f"Prepared data file not found: {data_path}\n"
        "Run 02_prepare_data.py first."
    )


# ==========================================
# 3. LOAD DATA
# ==========================================

edges = pd.read_csv(edges_path)
data = pd.read_csv(data_path)

print("\n========== DATA LOADED ==========")
print("Edges shape:", edges.shape)
print("Prepared labelled data shape:", data.shape)

print("\nEdge columns:", edges.columns.tolist())
print("Prepared data columns:", data.columns.tolist())


# ==========================================
# 4. VALIDATE REQUIRED COLUMNS
# ==========================================

required_edge_columns = {"txId1", "txId2"}
required_data_columns = {"txId", "target"}

if not required_edge_columns.issubset(edges.columns):
    raise ValueError(
        "Edge file must contain columns: txId1 and txId2."
    )

if not required_data_columns.issubset(data.columns):
    raise ValueError(
        "Prepared data must contain columns: txId and target."
    )


# ==========================================
# 5. BUILD TRANSACTION GRAPH
# ==========================================

print("\n========== BUILDING TRANSACTION GRAPH ==========")

graph = nx.from_pandas_edgelist(
    edges,
    source="txId1",
    target="txId2",
    create_using=nx.DiGraph()
)

print("Graph created successfully!")
print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())
print("Is directed:", graph.is_directed())


# ==========================================
# 6. ATTACH KNOWN TRANSACTION LABELS
# ==========================================

# target = 0 means licit
# target = 1 means illicit

labels = data.set_index("txId")["target"].to_dict()

nx.set_node_attributes(
    graph,
    labels,
    name="target"
)

labelled_nodes = sum(
    1
    for _, attributes in graph.nodes(data=True)
    if "target" in attributes
)

print("\n========== NODE LABEL INFORMATION ==========")
print("Nodes with known labels:", labelled_nodes)

print(
    "Nodes without known labels:",
    graph.number_of_nodes() - labelled_nodes
)


# ==========================================
# 7. BASIC GRAPH STATISTICS
# ==========================================

print("\n========== GRAPH STATISTICS ==========")

undirected_graph = graph.to_undirected()

connected_components = nx.number_connected_components(
    undirected_graph
)

print("Connected components:", connected_components)


# ==========================================
# 8. SAVE GRAPH SUMMARY
# ==========================================

summary = pd.DataFrame([{
    "nodes": graph.number_of_nodes(),
    "edges": graph.number_of_edges(),
    "labelled_nodes": labelled_nodes,
    "unlabelled_nodes": (
        graph.number_of_nodes() - labelled_nodes
    ),
    "is_directed": graph.is_directed(),
    "connected_components": connected_components
}])

summary.to_csv(output_path, index=False)

print("\n========== GRAPH SUMMARY SAVED ==========")
print("Saved to:", output_path)
print(summary.to_string(index=False))

print("\nTransaction graph analysis completed successfully!")
