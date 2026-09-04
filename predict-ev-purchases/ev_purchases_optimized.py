import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from scipy.stats import rankdata
import warnings, gc
warnings.filterwarnings("ignore")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
orig = pd.read_csv("original/EV_Adoption_and_Range_Anxiety_Dataset.csv")

TARGET = "Will_Buy_EV"
train[TARGET] = train[TARGET].map({"Yes":1, "No":0})
orig[TARGET] = orig[TARGET].map({"Yes":1, "No":0})

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Columns: {list(train.columns)}")

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan

combined = pd.concat([train, test], ignore_index=True)
combined.drop(columns=["Number_of_Cars_Owned"], inplace=True, errors="ignore")

cat_cols = combined.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in combined.columns if c not in cat_cols + ["id", "is_train", TARGET]]

print(f"Categorical: {cat_cols}")
print(f"Numerical: {num_cols}")

print("Extracting 8 digit levels...")
for c in num_cols:
    for k in [-1, 0, 1, 2, 3, 4, 5, 6]:
        combined[f"{c}_d{k}"] = (combined[c].fillna(0) // (10**k) % 10).astype("int8")

print("Adding CTGAN artifact flags...")
combined["is_30k_spike"] = (combined["Annual_Income_USD"] == 30000.0).astype("int8")
combined["is_millionaire_cliff"] = (combined["Annual_Income_USD"] >= 170537.0).astype("int8")
combined["is_dead_zone"] = ((combined["Annual_Income_USD"] >= 38000.0) &
                            (combined["Annual_Income_USD"] <= 420000)).astype("int8")
combined['is_env_hater'] = (combined['Environmental_Concern_Level'] == 1).astype('int8')
combined['income_per_km'] = ((combined['Annual_Income_USD'] / (combined['Daily_Commute_km'] + 1))
                             .astype('float32'))
combined['charge_ratio'] = (combined['Charging_Stations_Near_Home'] / (combined['Charging_Stations_Near_Work'] + 1)).astype('float32')
combined['age_income'] = (combined['Age'] * combined['Annual_Income_USD'] / 1e6).astype('float32')

print("Adding smooth keys...")
combined['income_int'] = np.floor(combined['Annual_Income_USD']).astype(str)
combined['income100'] = np.floor(combined['Annual_Income_USD'] / 100.0).astype(str)
combined['income1000'] = np.floor(combined['Annual_Income_USD'] / 1000.0).astype(str)
combined['commute_int'] = np.floor(combined['Daily_Commute_km']).astype(str)
combined['commute10'] = np.floor(combined['Daily_Commute_km'] / 10.0).astype(str)
combined['age_int'] = np.floor(combined['Age']).astype(str)
smooth_cats = ['income_int', 'income100', 'income1000', 'commute_int', 'commute10', 'age_int']

print("Mapping original dataset means + std...")
orig_mean = orig[TARGET].mean()
for col in cat_cols:
    if col in orig.columns:
        stats = orig.groupby(col, observed=False)[TARGET].mean()
        combined[f"{col}_org_mean"] = combined[col].map(stats).fillna(orig_mean).astype("float32")
        stats_std = orig.groupby(col, observed=False)[TARGET].std()
        combined[f"{col}_org_std"] = combined[col].map(stats_std).fillna(0).astype("float32")

print("Frequency + Count encoding...")
all_te_cols = cat_cols + smooth_cats
for col in all_te_cols:
    freq = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_fe"] = combined[col].map(freq).astype("float32").fillna(0)
    cnt = combined[col].value_counts().to_dict()
    combined[f"{col}_cnt"] = combined[col].map(cnt).astype('int32').fillna(0)

train_df = combined[combined['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
test_df = combined[combined['is_train'] == 0].drop(columns=['is_train', TARGET]).reset_index(drop=True)
del combined; gc.collect()

eval_cols = [c for c in train_df.columns if c not in ['id', TARGET] and pd.api.types.is_numeric_dtype(train_df[c])]
corr = train_df[eval_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] == 1.0)]
drop_const = [c for c in eval_cols if train_df[c].nunique() <= 1]
DROP = list(set(drop_corr + drop_const))

train_df.drop(columns=DROP, inplace=True, errors='ignore')
test_df.drop(columns=DROP, inplace=True, errors='ignore')
del corr, upper; gc.collect()

FEATURES = [c for c in test_df.columns if c != 'id']
TE_COLS = [c for c in all_te_cols if c in FEATURES]

print(f"Dropped {len(DROP)} features")
print(f"Final features: {len(FEATURES)}, TE columns: {len(TE_COLS)}")

Folds = 5
X = train_df[FEATURES]
y = train_df[TARGET]
X_test = test_df[FEATURES]
test_ids = test_df['id'].values

skf = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

print(f"\n{'=' * 50}")
print("Training: XGBOOST (Full Features)")
print(f"{'=' * 50}")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx].copy()
    y_val = y.iloc[val_idx]
    X_te = X_test.copy()

    target_encoder1 = TargetEncoder(shuffle=True, cv=5, smooth="auto", random_state=42)
    target_encoder2 = TargetEncoder(shuffle=True, cv=5, smooth=10.0, random_state=123)
    target_encoder4 = TargetEncoder(shuffle=True, cv=5, smooth=50.0, random_state=456)

    encoder1_train = target_encoder1.fit_transform(X_train[TE_COLS], y_train)
    encoder1_val = target_encoder1.transform(X_val[TE_COLS])
    encoder1_test = target_encoder1.transform(X_te[TE_COLS])

    encoder2_train = target_encoder2.fit_transform(X_train[TE_COLS], y_train)
    encoder2_val = target_encoder2.transform(X_val[TE_COLS])
    encoder2_test = target_encoder2.transform(X_te[TE_COLS])

    encoder4_train = target_encoder4.fit_transform(X_train[TE_COLS], y_train)
    encoder4_val = target_encoder4.transform(X_val[TE_COLS])
    encoder4_test = target_encoder4.transform(X_te[TE_COLS])

    for i, col in enumerate(TE_COLS):
        X_train[f"{col}_TE1"] = encoder1_train[:, i].astype('float32')
        X_val[f"{col}_TE1"] = encoder1_val[:, i].astype('float32')
        X_te[f"{col}_TE1"] = encoder1_test[:, i].astype('float32')
        X_train[f"{col}_TE2"] = encoder2_train[:, i].astype('float32')
        X_val[f"{col}_TE2"] = encoder2_val[:, i].astype('float32')
        X_te[f"{col}_TE2"] = encoder2_test[:, i].astype('float32')
        X_train[f"{col}_TE4"] = encoder4_train[:, i].astype('float32')
        X_val[f"{col}_TE4"] = encoder4_val[:, i].astype('float32')
        X_te[f"{col}_TE4"] = encoder4_test[:, i].astype('float32')

        X_train.drop(columns=[col], inplace=True)
        X_val.drop(columns=[col], inplace=True)
        X_te.drop(columns=[col], inplace=True)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.02+0.01,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.2+0.1,
        "tree_method": "hist",
        "random_state": 42 + fold,
        "n_jobs": -1
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_te)

    model = xgb.train(params, dtrain, num_boost_round=10000, evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      early_stopping_rounds=300, verbose_eval=1000)
    val_predictions = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
    test_preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    auc = roc_auc_score(y_val, val_predictions)
    oof_predictions[val_idx] = val_predictions
    test_predictions += test_preds / Folds
    del X_train, X_val, X_te, dtrain, dvalid, dtest
    gc.collect()
oof_auc = roc_auc_score(y, oof_predictions)
print(f"\n  >>> XGBOOST OOF AUC: {oof_auc:.6f}")

sub = pd.DataFrame({
    "id": test_ids,
    "Will_Buy_EV": test_predictions
})
sub.to_csv("submission.csv", index=False)
print("Submission saved!")
print(sub.head())