import pandas as pd


# 1. Load transaction data
print("Loading transaction data...")

df = pd.read_csv(
    "data/raw/PaySim.csv",
    nrows=500000
)


# 2. Load suspicious accounts detected by the graph detector
suspicious_accounts = pd.read_csv(
    "results/suspicious_accounts.csv"
)

suspicious_set = set(suspicious_accounts["account"])


# 3. Identify transactions involving suspicious accounts
df["involves_suspicious_account"] = (
    df["nameOrig"].isin(suspicious_set)
    | df["nameDest"].isin(suspicious_set)
)


# 4. Evaluate against PaySim fraud labels
true_fraud = df["isFraud"] == 1

detected = df["involves_suspicious_account"]

true_positives = (true_fraud & detected).sum()
false_positives = (~true_fraud & detected).sum()
false_negatives = (true_fraud & ~detected).sum()

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


# 5. Display results
print("\n===== GRAPH DETECTOR EVALUATION =====")

print("True positives:", true_positives)
print("False positives:", false_positives)
print("False negatives:", false_negatives)

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))


# 6. Save results
df.to_csv(
    "results/graph_evaluation.csv",
    index=False
)

print("\nEvaluation saved to results/graph_evaluation.csv")