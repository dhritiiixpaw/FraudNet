import pandas as pd
from pathlib import Path


# 1. Load transaction data
print("Loading transaction data...")

df = pd.read_csv(
    "data/raw/PaySim.csv",
    nrows=500000
)


# 2. Load the trained-model prediction data
print("Loading model predictions...")

# Recreate the Random Forest predictions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

features = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X = df[features]
y = df["isFraud"]

X = pd.get_dummies(X, columns=["type"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)


# 3. Get transactions flagged by the model
test_indices = y_test.index

flagged_indices = test_indices[predictions == 1]

flagged_transactions = df.loc[flagged_indices].copy()


# 4. Generate sample reports
print("\nGenerating sample reports...")

reports = []

for report_number, (_, row) in enumerate(
    flagged_transactions.iterrows(),
    start=1
):

    report = {
        "report_id": f"FRAUDNET-SAR-{report_number:04d}",
        "transaction_id": row["nameOrig"],
        "transaction_type": row["type"],
        "amount": row["amount"],
        "sender_account": row["nameOrig"],
        "receiver_account": row["nameDest"],
        "timestamp_step": row["step"],
        "model_flagged": True,
        "actual_fraud_label": row["isFraud"],
        "report_status": "Pending review",
        "suspicion_reason": (
            "Transaction flagged by FraudNet Random Forest model "
            "for further investigation."
        )
    }

    reports.append(report)


# 5. Save reports
reports_df = pd.DataFrame(reports)

Path("results").mkdir(exist_ok=True)

reports_df.to_csv(
    "results/sample_sar_reports.csv",
    index=False
)


# 6. Print summary
print("\n===== FRAUDNET REPORT SUMMARY =====")

print("Reports generated:", len(reports_df))

print(
    "Reports involving known fraud:",
    reports_df["actual_fraud_label"].sum()
)

print(
    "Reports requiring review:",
    len(reports_df)
)

print("\nSample report:")

if not reports_df.empty:
    print(reports_df.iloc[0].to_string())

print("\nReports saved to results/sample_sar_reports.csv")