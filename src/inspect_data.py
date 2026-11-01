print("Script started!")

import pandas as pd

def inspect_data():
    print("Function called!")

    df = pd.read_csv("data/raw/PaySim.csv", nrows=10)

    print("CSV loaded!")
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df["isFraud"].value_counts())
#inspect_data()
def count_fraud():
    fraud_counts = {0: 0, 1: 0}

    for chunk in pd.read_csv("data/raw/PaySim.csv", chunksize=100000):
        counts = chunk["isFraud"].value_counts()

        fraud_counts[0] += counts.get(0, 0)
        fraud_counts[1] += counts.get(1, 0)

    print(fraud_counts)

#count_fraud()
def fraud_by_type():
    fraud_data = pd.DataFrame()

    for chunk in pd.read_csv("data/raw/PaySim.csv", chunksize=100000):
        counts = chunk.groupby(["type", "isFraud"]).size().unstack(fill_value=0)

        fraud_data = fraud_data.add(counts, fill_value=0)

    print(fraud_data)
#fraud_by_type()
def inspect_fraud_examples():
    fraud_examples = []

    for chunk in pd.read_csv("data/raw/PaySim.csv", chunksize=100000):
        fraud = chunk[chunk["isFraud"] == 1]

        fraud_examples.append(
            fraud[["step", "type", "amount", "nameOrig", "nameDest"]]
        )

        if sum(len(data) for data in fraud_examples) >= 20:
            break

    examples = pd.concat(fraud_examples)

    print(examples.head(20))
#inspect_fraud_examples()
def count_fraud_accounts():
    fraud_senders = set()

    for chunk in pd.read_csv("data/raw/PaySim.csv", chunksize=100000):
        fraud = chunk[chunk["isFraud"] == 1]

        fraud_senders.update(fraud["nameOrig"])

    print("Unique fraud sender accounts:", len(fraud_senders))
#count_fraud_accounts()
def inspect_fraud_accounts():
    fraud_senders = set()
    fraud_receivers = set()

    # First pass: collect accounts involved in fraud
    for chunk in pd.read_csv("data/raw/PaySim.csv", chunksize=100000):
        fraud = chunk[chunk["isFraud"] == 1]

        fraud_senders.update(fraud["nameOrig"])
        fraud_receivers.update(fraud["nameDest"])

    fraud_accounts = fraud_senders | fraud_receivers

    print("Unique fraud senders:", len(fraud_senders))
    print("Unique fraud receivers:", len(fraud_receivers))
    print("Accounts appearing as both:", len(fraud_senders & fraud_receivers))

    # Second pass: count transactions involving those accounts
    transaction_count = 0

    for chunk in pd.read_csv("data/raw/PaySim.csv", chunksize=100000):
        involved = (
            chunk["nameOrig"].isin(fraud_accounts)
            | chunk["nameDest"].isin(fraud_accounts)
        )

        transaction_count += involved.sum()

    print("Transactions involving fraud accounts:", transaction_count)


inspect_fraud_accounts()