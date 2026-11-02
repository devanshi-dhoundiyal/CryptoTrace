
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "elliptic_bitcoin_dataset"
RESULTS = ROOT / "results"

PREDICTIONS_PATH = RESULTS / "xgboost_predictions.csv"
EDGES_PATH = DATASET / "elliptic_txs_edgelist.csv"
FEATURES_PATH = DATASET / "elliptic_txs_features.csv"

# --------------------------------------------------
# 2. API setup
# --------------------------------------------------

app = FastAPI(
    title="CryptoTrace API",
    description="Transaction prediction and graph investigation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 3. Load the data once when the API starts
# --------------------------------------------------

predictions_df = pd.read_csv(PREDICTIONS_PATH)
edges_df = pd.read_csv(EDGES_PATH)
features_df = pd.read_csv(
    FEATURES_PATH,
    header=None,
    usecols=[0, 1],
    names=["txId", "time_step"],
)

for frame, column in [
    (predictions_df, "txId"),
    (edges_df, "txId1"),
    (edges_df, "txId2"),
    (features_df, "txId"),
]:
    frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")

predictions_df["predicted_class"] = predictions_df[
    "predicted_class"
].astype(int)

predictions_df["illicit_probability"] = pd.to_numeric(
    predictions_df["illicit_probability"],
    errors="raise",
)

predictions_df["time_step"] = predictions_df["time_step"].astype(int)

# Ensure probabilities are valid.
if not predictions_df["illicit_probability"].between(0, 1).all():
    raise ValueError("Illicit probabilities must be between 0 and 1.")

if predictions_df["txId"].duplicated().any():
    raise ValueError("Duplicate transaction IDs found in predictions.")

# The API must never invent predictions for unscored transactions.
prediction_lookup = predictions_df.set_index("txId").to_dict("index")

time_lookup = features_df.drop_duplicates("txId").set_index(
    "txId"
)["time_step"].to_dict()

# Build a transaction adjacency index for graph exploration.
adjacency = {}

for source, target in edges_df[["txId1", "txId2"]].itertuples(
    index=False,
    name=None,
):
    adjacency.setdefault(source, set()).add(target)
    adjacency.setdefault(target, set()).add(source)

print(f"Loaded predictions: {len(predictions_df):,}")
print(f"Loaded transaction edges: {len(edges_df):,}")
print(f"Transactions with feature records: {len(time_lookup):,}")

# --------------------------------------------------
# 4. Helpers
# --------------------------------------------------

def make_transaction(tx_id: int) -> dict:
    prediction = prediction_lookup.get(tx_id)

    if prediction is None:
        return {
            "id": str(tx_id),
            "time_step": time_lookup.get(tx_id),
            "has_prediction": False,
            "predicted_class": None,
            "illicit_probability": None,
            "risk": "Unknown",
        }

    probability = float(prediction["illicit_probability"])

    # These are interface display bands, not calibrated risk categories.
    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.30:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "id": str(tx_id),
        "time_step": int(prediction["time_step"]),
        "has_prediction": True,
        "predicted_class": int(prediction["predicted_class"]),
        "illicit_probability": probability,
        "risk": risk,
    }


# --------------------------------------------------
# 5. Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "CryptoTrace API is running",
        "docs": "/docs",
    }


@app.get("/api/summary")
def summary():
    total_predictions = len(predictions_df)

    return {
        "dataset_transactions": int(len(features_df)),
        "labelled_prediction_rows": total_predictions,
        "graph_connections": int(len(edges_df)),
        "predicted_illicit": int(
            (predictions_df["predicted_class"] == 1).sum()
        ),
        "predicted_licit": int(
            (predictions_df["predicted_class"] == 0).sum()
        ),
        "prediction_coverage": "Labelled test transactions only",
    }


@app.get("/api/transactions")
def list_transactions(
    search: str = "",
    limit: int = Query(default=50, ge=1, le=200),
    high_risk_only: bool = False,
):
    df = predictions_df

    if search.strip():
        df = df[
            df["txId"].astype(str).str.contains(
                search.strip(),
                regex=False,
            )
        ]

    if high_risk_only:
        df = df[df["illicit_probability"] >= 0.70]

    df = df.sort_values("illicit_probability", ascending=False).head(limit)

    return {
        "count": int(len(df)),
        "transactions": [
            make_transaction(int(tx_id))
            for tx_id in df["txId"].tolist()
        ],
    }


@app.get("/api/transactions/{tx_id}")
def get_transaction(tx_id: int):
    if tx_id not in time_lookup:
        raise HTTPException(status_code=404, detail="Transaction not found")

    transaction = make_transaction(tx_id)

    neighbors = sorted(adjacency.get(tx_id, set()))

    transaction["neighbors"] = [
        make_transaction(neighbor_id)
        for neighbor_id in neighbors[:100]
    ]

    transaction["neighbor_count"] = len(neighbors)

    return transaction


@app.get("/api/graph")
def get_graph(
    seed: int | None = None,
    limit: int = Query(default=40, ge=1, le=100),
):
    # If no seed is specified, begin with the highest predicted
    # illicit probabilities and explore their recorded neighbors.
    if seed is None:
        seeds = (
            predictions_df.sort_values(
                "illicit_probability",
                ascending=False,
            )["txId"]
            .head(min(10, limit))
            .astype(int)
            .tolist()
        )
    else:
        if seed not in time_lookup:
            raise HTTPException(
                status_code=404,
                detail="Seed transaction not found",
            )
        seeds = [seed]

    selected_ids = set(seeds)

    for tx_id in seeds:
        for neighbor_id in sorted(adjacency.get(tx_id, set())):
            if len(selected_ids) >= limit:
                break
            selected_ids.add(neighbor_id)

        if len(selected_ids) >= limit:
            break

    # Include real, recorded edges between the displayed nodes.
    graph_edges = edges_df[
        edges_df["txId1"].isin(selected_ids)
        & edges_df["txId2"].isin(selected_ids)
    ]

    nodes = [
        make_transaction(tx_id)
        for tx_id in sorted(selected_ids)
    ]

    edges = [
        {
            "id": f"{int(row.txId1)}-{int(row.txId2)}",
            "source": str(int(row.txId1)),
            "target": str(int(row.txId2)),
        }
        for row in graph_edges.itertuples(index=False)
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "seed_transactions": [str(tx_id) for tx_id in seeds],
        "note": (
            "Edges are recorded transaction relationships. "
            "Transactions without saved XGBoost predictions have "
            "unknown prediction status. Display risk bands are "
            "illustrative thresholds, not calibrated probabilities."
        ),
    }
