import pandas as pd
import argparse
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train(
        model_name: str = "predict_stellar_class",
        seed: int = 2026,
        train: bool = True
):
    model_path = f"{model_name}.joblib"
    df = pd.read_csv("data/train.csv")

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    y = df["class"]
    X = df.drop(columns=["id", "class"])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                        random_state=seed, stratify=y)

    numeric_features = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "u_g", "g_r", "r_i", "i_z"]
    categorical_features = ["spectral_type", "galaxy_population"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features
            ),
            (
                "cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features
            )
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, predictions))
    print(classification_report(y_val, predictions))

    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

def predict_stellar_class(
    model_name: str = "predict_stellar_class",
    seed: int = 2026,
    train: bool = False
):
    model = joblib.load(f"{model_name}.joblib")
    test_df = pd.read_csv("data/test.csv")

    test_df["u_g"] = test_df["u"] - test_df["g"]
    test_df["g_r"] = test_df["g"] - test_df["r"]
    test_df["r_i"] = test_df["r"] - test_df["i"]
    test_df["i_z"] = test_df["i"] - test_df["z"]

    test_ids = test_df["id"]
    X_test = test_df.drop(columns=["id"])

    predictions = model.predict(X_test)
    submission = pd.DataFrame({
        "id": test_ids,
        "class": predictions
    })
    submission.to_csv("submission.csv", index=False)
    print(submission.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=2026)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)
    else:
        predict_stellar_class(**args)