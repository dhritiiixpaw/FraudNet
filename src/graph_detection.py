import pandas as pd
import networkx as nx
from pathlib import Path


# 1. Load transaction data
print("Loading transaction data...")

df = pd.read_csv(
    "data/raw/PaySim.csv",
    nrows=500000
)

print("Dataset loaded:", df.shape)


# 2. Build a graph using transfer transactions
print("\nBuilding account network...")

transfers = df[df["type"] == "TRANSFER"].copy()

G = nx.DiGraph()

for _, row in transfers.iterrows():
    sender = row["nameOrig"]
    receiver = row["nameDest"]

    G.add_edge(sender, receiver)

print("Transfer transactions:", len(transfers))
print("Accounts in network:", G.number_of_nodes())
print("Connections in network:", G.number_of_edges())


# 3. Identify accounts involved in known fraud transactions
print("\nIdentifying accounts involved in fraud...")

fraud_transactions = df[df["isFraud"] == 1]

fraud_accounts = set(fraud_transactions["nameOrig"])
fraud_accounts.update(fraud_transactions["nameDest"])

print("Known fraud transactions:", len(fraud_transactions))
print("Accounts involved in known fraud:", len(fraud_accounts))


# 4. Find transactions connected to those accounts
print("\nFinding connected transactions...")

connected_transactions = df[
    df["nameOrig"].isin(fraud_accounts)
    | df["nameDest"].isin(fraud_accounts)
].copy()

connected_transactions["graph_flag"] = 1

print(
    "Transactions connected to fraud accounts:",
    len(connected_transactions)
)


# 5. Mark all other transactions as not graph-flagged
df["graph_flag"] = (
    df["nameOrig"].isin(fraud_accounts)
    | df["nameDest"].isin(fraud_accounts)
).astype(int)


# 6. Evaluate graph flags against known labels
true_positives = (
    (df["graph_flag"] == 1) & (df["isFraud"] == 1)
).sum()

false_positives = (
    (df["graph_flag"] == 1) & (df["isFraud"] == 0)
).sum()

false_negatives = (
    (df["graph_flag"] == 0) & (df["isFraud"] == 1)
).sum()

precision = (
    true_positives / (true_positives + false_positives)
    if true_positives + false_positives > 0
    else 0
)

recall = (
    true_positives / (true_positives + false_negatives)
    if true_positives + false_negatives > 0
    else 0
)


# 7. Print evaluation
print("\n===== GRAPH DETECTOR RESULTS =====")

print("True positives:", true_positives)
print("False positives:", false_positives)
print("False negatives:", false_negatives)

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))


# 8. Save flagged transactions
Path("results").mkdir(exist_ok=True)

connected_transactions.to_csv(
    "results/graph_flagged_transactions.csv",
    index=False
)

print("\nFlagged transactions saved to:")
print("results/graph_flagged_transactions.csv")