import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

df = pd.read_csv("data/train.csv")

print(f"Shape: {df.shape}")
print(df.head())
print(df["Will_Buy_EV"].value_counts())

df = df.drop(columns=["id"])

# df["Income_Per_Car"] = (
#     df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"] + 1)
# )
# df["Charging_Access_Score"] = (df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"])
# df["Commute_Charging_Ratio"] = (
#     df["Daily_Commute_km"] / (df["Charging_Access_Score"] + 1)
# )
# df["Commute_Environmental_Index"] = (df["Daily_Commute_km"] * df["Environmental_Concern_Level"])
# df["Commute_Charging_Index"] = (df["Daily_Commute_km"] * (df["Charging_Access_Score"] + 1))
# df["Income_Age_Index"] = (df["Annual_Income_USD"] * df["Age"])

df["Will_Buy_EV"] = (df["Will_Buy_EV"].map({"Yes": 1, "No": 0}).astype("int8"))

X = df.drop(columns=["Will_Buy_EV"])
y = df["Will_Buy_EV"]

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

for col in categorical_features:
    X[col] = X[col].astype("category")
for col in numerical_features:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    if "Missing" not in X[col].cat.categories:
        X[col] = X[col].cat.add_categories(["Missing"])
    X[col] = X[col].fillna("Missing")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

print(f"Negative: {negative}")
print(f"Positive: {positive}")
print(f"Scale positive weight: {scale_pos_weight}")

model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=(0.01 + 0.02),
    num_leaves=63,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary",
    metric="auc",
    n_jobs=-1,
    random_state=42,
    verbosity=-1
)

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=["train", "validation"],
    callbacks=[early_stopping(stopping_rounds=100, verbose=True), log_evaluation(period=50)]
)

y_probability = model.predict_proba(X_val)[:, 1]
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

print(
    "\nBest iteration:",
    model.best_iteration_
)

print(
    "Best validation AUC:",
    model.best_score_["validation"]["auc"]
)

test_df = pd.read_csv("./data/test.csv")

print(f"\nTest shape: {test_df.shape}")
print(test_df.head())

test_ids = test_df["id"].copy()
test_df = test_df.drop(columns=["id"])

X_test = test_df.copy()
for col in categorical_features:
    X_test[col] = pd.Categorical(
        X_test[col],
        categories=X[col].cat.categories,
    )
    X_test[col] = X_test[col].fillna("Missing")

for col in numerical_features:
    median_value = X[col].median()
    X_test[col] = X_test[col].fillna(median_value)

test_probability = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_probability})

submission.to_csv("submission.csv", index=False)

# test_df["Income_Per_Car"] = (
#     test_df["Annual_Income_USD"] / (test_df["Number_of_Cars_Owned"] + 1)
# )
# test_df["Charging_Access_Score"] = (test_df["Charging_Stations_Near_Home"] + test_df["Charging_Stations_Near_Work"])
# test_df["Commute_Charging_Ratio"] = (
#     test_df["Daily_Commute_km"] / (test_df["Charging_Access_Score"] + 1)
# )
# test_df["Commute_Environmental_Index"] = (test_df["Daily_Commute_km"] * test_df["Environmental_Concern_Level"])
# test_df["Commute_Charging_Index"] = (test_df["Daily_Commute_km"] * (test_df["Charging_Access_Score"] + 1))
# test_df["Income_Age_Index"] = (test_df["Annual_Income_USD"] * test_df["Age"])