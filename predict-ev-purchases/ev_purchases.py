import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from xgboost import XGBClassifier

df = pd.read_csv("data/train.csv")

print(f"Shape: {df.shape}")
print(df.head())
print(df["Will_Buy_EV"].value_counts())

df = df.drop(columns=["id"])

df["Income_Per_Car"] = (
    df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"] + 1)
)
df["Charging_Access_Score"] = (df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"])
df["Commute_Charging_Ratio"] = (
    df["Daily_Commute_km"] / (df["Charging_Access_Score"] + 1)
)
df["Commute_Environmental_Index"] = (df["Daily_Commute_km"] * df["Environmental_Concern_Level"])
df["Commute_Charging_Index"] = (df["Daily_Commute_km"] * (df["Charging_Access_Score"] + 1))
df["Income_Age_Index"] = (df["Annual_Income_USD"] * df["Age"])

df["Will_Buy_EV"] = (df["Will_Buy_EV"].map({"Yes": 1, "No": 0}))

X = df.drop(columns=["Will_Buy_EV"])
y = df["Will_Buy_EV"]

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

print(f"\nProcessed feature shape: {X_train_processed.shape}")

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

print(f"Negative: {negative}")
print(f"Positive: {positive}")
print(f"Scale positive weight: {scale_pos_weight}")

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=(0.01 + 0.02),
    max_depth=7,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train,
    eval_set=[
        (X_train_processed, y_train),
        (X_val_processed, y_val),
    ],
    verbose=50
)

y_probability = model.predict_proba(X_val_processed)[:, 1]

thresholds = np.arange(0.1, 0.91, 0.01)
best_threshold = 0
best_f1 = 0
for threshold in thresholds:
    y_prediction = (y_probability >= threshold).astype(int)
    f1 = f1_score(
        y_val,
        y_prediction,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
y_prediction = (y_probability >= 0.5).astype(int)

accuracy = accuracy_score(y_val, y_prediction)
precision = precision_score(y_val, y_prediction, zero_division=0)
recall = recall_score(y_val, y_prediction, zero_division=0)
f1 = f1_score(y_val, y_prediction, zero_division=0)
auc = roc_auc_score(y_val, y_probability)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUC: {auc:.4f}")

print("\nClassification report:")
print(classification_report(y_val, y_prediction, target_names=["No", "Yes"]))

test_df = pd.read_csv("./data/test.csv")

print(f"\nTest shape: {test_df.shape}")
print(test_df.head())

test_ids = test_df["id"].copy()
test_df = test_df.drop(columns=["id"])

test_df["Income_Per_Car"] = (
    test_df["Annual_Income_USD"] / (test_df["Number_of_Cars_Owned"] + 1)
)
test_df["Charging_Access_Score"] = (test_df["Charging_Stations_Near_Home"] + test_df["Charging_Stations_Near_Work"])
test_df["Commute_Charging_Ratio"] = (
    test_df["Daily_Commute_km"] / (test_df["Charging_Access_Score"] + 1)
)
test_df["Commute_Environmental_Index"] = (test_df["Daily_Commute_km"] * test_df["Environmental_Concern_Level"])
test_df["Commute_Charging_Index"] = (test_df["Daily_Commute_km"] * (test_df["Charging_Access_Score"] + 1))
test_df["Income_Age_Index"] = (test_df["Annual_Income_USD"] * test_df["Age"])

X_test_processed = preprocessor.transform(test_df)
print(f"Processed test shape: {X_test_processed.shape}")

test_probability = model.predict_proba(X_test_processed)[:, 1]
submission = pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_probability})

submission.to_csv("submission.csv", index=False)