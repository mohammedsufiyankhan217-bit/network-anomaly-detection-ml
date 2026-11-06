
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("dataset.csv", low_memory=False)

# Clean data
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Keep BENIGN and DDoS traffic
df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

# Separate features and target
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=["number"])
y = df["Label"].map({"BENIGN": 0, "DDoS": 1})

# Remove invalid rows and constant features
X = X.replace([np.inf, -np.inf], np.nan)
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows]
X = X.loc[:, X.nunique() > 1]

# Split data into training and testing sets
# Keep identical feature rows in the same split
groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(
    "Overlapping feature rows:",
    len(
        set(pd.util.hash_pandas_object(X_train, index=False))
        & set(pd.util.hash_pandas_object(X_test, index=False))
    )
)

print("Training label distribution:")
print(y_train.value_counts())

print("Testing label distribution:")
print(y_test.value_counts())

print("Training Random Forest model...")

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)
# Display the top 10 most important features
importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Evaluate the model
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nMODEL EVALUATION")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")


print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["BENIGN", "DDoS"],
    output_dict=True,
    zero_division=0
)

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["BENIGN", "DDoS"],
    zero_division=0
))



print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and feature names



joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).tolist(),
        "report": report
    },
    "network_model.pkl"
)




print("\nModel saved as network_model.pkl")
print("Training completed successfully!")
