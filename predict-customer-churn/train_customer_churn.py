import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from datasets.customer_churn_dataset import CustomerChurnDataset
from models import load_model, save_model
import joblib


def train(
    model_name: str = "predict_customer_churn",
    num_epoch: int = 50,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 2026,
    weight_decay: float = None,
    train: bool = True
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU instead")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = pd.read_csv("data/train.csv")
    train_add_data = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master"
                                 "/data/Telco-Customer-Churn.csv")

    id_col = ["id", "customerID"]
    df = pd.concat([train_data, train_add_data], axis=0)
    df.drop(columns=id_col, inplace=True)

    xcols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV',
             'StreamingMovies']
    df["all_same_services"] = df[xcols].nunique(axis=1) == 1
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.reset_index(drop=True)

    cat_cols = df.describe(include="object").columns[:-1]
    joblib.dump(cat_cols, "cat_cols.joblib")

    df["Churn"] = df['Churn'].map({'No':0, 'Yes':1})
    df.drop_duplicates(inplace=True)
    df["dominance_of_nointernetservice"] = (df[xcols] == "No internet service").all(axis=1)

    df["is_fixed_contract_time"] = df["Contract"].apply(lambda x: True if "year" in x else False)
    df["is_automatic_payment"] = df["PaymentMethod"].apply(lambda x: True if "automatic" in x else False)

    encodings = {}
    target = "Churn"
    for col in cat_cols:
        mapping = df.groupby(col)[target].mean()
        encodings[col] = mapping
    joblib.dump(encodings, "encodings.joblib")

    for col in cat_cols:
        df[col] = df[col].map(encodings[col])

    scaler = StandardScaler()
    scale_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'all_same_services',
       'is_fixed_contract_time', 'is_automatic_payment',
       'dominance_of_nointernetservice']
    joblib.dump(scale_cols, "scale_cols.joblib")

    df["charge_tenure_interaction"] = (df["TotalCharges"] * df["MonthlyCharges"] * df["tenure"])
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    joblib.dump(scaler, "scaler.joblib")

    df_0 = df[df["Churn"] == 0]
    df_1 = df[df["Churn"] == 1]

    df_0_down = resample(df_0, replace=False, n_samples=150000, random_state=seed)
    df = pd.concat([df_0_down, df_1])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.dropna()

    target_col = "Churn"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    train_dataset = CustomerChurnDataset(X_train, y_train)
    val_dataset = CustomerChurnDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, with_weights=False, num_features=X_train.shape[1])
    model.to(device)
    model.train()

    loss_func = torch.nn.CrossEntropyLoss()
    if weight_decay is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        metrics = {"train_acc": [], "val_acc": [], "train_loss": 0.0, "val_loss": 0.0}
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            out = model(data)

            optimizer.zero_grad()
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out, 1)
            matched = torch.sum(torch.eq(predicted, labels))
            metrics["train_acc"].append(matched.item() / len(labels))
            metrics["train_loss"] += loss.item()
        with torch.inference_mode():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                out = model(data)
                loss = loss_func(out, labels)

                _, predicted = torch.max(out, 1)
                matched = torch.sum(torch.eq(predicted, labels))
                metrics["val_acc"].append(matched.item() / len(labels))
                metrics["val_loss"] += loss.item()

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        epoch_train_loss = metrics["train_loss"] / len(train_loader)
        epoch_val_loss = metrics["val_loss"] / len(val_loader)

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_acc={epoch_val_acc:.4f} "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f} "
        )
    save_model(model)

def test(model_name: str = "predict_customer_churn",
    num_epoch: int = 50,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 2026,
    weight_decay: float = None,
    train: bool = False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU instead")
        device = torch.device("cpu")

    test_data = pd.read_csv("data/test.csv")

    id_col = ["id"]
    id_data = test_data["id"]
    test_data.drop(columns=id_col, inplace=True)

    xcols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
             'StreamingMovies']
    test_data["all_same_services"] = test_data[xcols].nunique(axis=1) == 1
    test_data["TotalCharges"] = pd.to_numeric(test_data["TotalCharges"], errors="coerce")
    test_data = test_data.reset_index(drop=True)

    cat_cols = joblib.load("cat_cols.joblib")

    test_data["dominance_of_nointernetservice"] = (test_data[xcols] == "No internet service").all(axis=1)
    test_data["is_fixed_contract_time"] = test_data["Contract"].apply(lambda x: True if "year" in x else False)
    test_data["is_automatic_payment"] = test_data["PaymentMethod"].apply(lambda x: True if "automatic" in x else False)

    encodings = joblib.load("encodings.joblib")
    for col in cat_cols:
        test_data[col] = test_data[col].map(encodings[col])

    test_data["charge_tenure_interaction"] = (test_data["TotalCharges"] * test_data["MonthlyCharges"]
                                              * test_data["tenure"])
    scaler = joblib.load("scaler.joblib")

    scale_cols = joblib.load("scale_cols.joblib")
    test_data[scale_cols] = scaler.transform(test_data[scale_cols])

    test_dataset = CustomerChurnDataset(test_data, id_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, with_weights=True, num_features=test_data.shape[1])
    model.to(device)
    model.eval()

    test_ids = []
    test_predictions = []
    with torch.inference_mode():
        for data, ids in test_loader:
            data = data.to(device)
            out = model(data)

            _, predicted = torch.max(out, 1)

            test_ids.extend(torch.tensor(ids, dtype=torch.long).tolist())
            labels = [0 if p == 0 else 1 for p in predicted.cpu()]
            test_predictions.extend(labels)
    df = pd.DataFrame({"id": test_ids, "Churn": test_predictions})
    df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=21)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)
    else:
        test(**args)