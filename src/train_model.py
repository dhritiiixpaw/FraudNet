import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# 1. Load the dataset
print("Loading dataset...")

df = pd.read_csv("data/raw/PaySim.csv", nrows=500000)

print("Dataset loaded.")
print("Dataset shape:", df.shape)


# 2. Select the features the models will use
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


# 3. Convert transaction type into numeric columns
X = pd.get_dummies(X, columns=["type"])

print("Features prepared.")


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# 5. Train the Decision Tree model
print("\nTraining Decision Tree...")

dt_model = DecisionTreeClassifier(
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)


# 6. Evaluate the Decision Tree
print("\n===== DECISION TREE RESULTS =====")

print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))


# 7. Train the Random Forest model
print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


# 8. Evaluate the Random Forest
print("\n===== RANDOM FOREST RESULTS =====")

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


print("\nBoth models finished training.")