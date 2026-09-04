import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets.EvPurchasesDataset import EvPurchasesDataset
from models import load_model, save_model


def train(
    model_name: str="predict_ev_purchases",
    num_epoch: int=50,
    lr: float=1e-2,
    batch_size: int=32,
    seed: int=2026,
    weight_decay: float=None,
    train: bool=True
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU instead")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = pd.read_csv("data/train.csv")
    print(f"Dataset shape: {train_data.shape}")
    print(train_data.head())
    print(train_data.dtypes)

    TARGET = "Will_Buy_EV"
    train_data = train_data.drop(columns=["id"])
    train_data = train_data.dropna(subset=[TARGET])

    train_data["Income_Per_Car"] = (train_data["Annual_Income_USD"] /
                                    (train_data["Number_of_Cars_Owned"] + 1))
    train_data["Commute_Environmental_Index"] = (train_data["Daily_Commute_km"] *
                                                 train_data["Environmental_Concern_Level"])
    train_data["Charging_Access_Score"] = (train_data["Charging_Stations_Near_Home"] +
                                           train_data["Charging_Stations_Near_Work"])
    train_data["Commute_Charging_Ratio"] = (train_data["Daily_Commute_km"] /
                                            (train_data["Charging_Access_Score"] + 1))
    train_data["Commute_Charging_Index"] = (train_data["Daily_Commute_km"] *
                                            (train_data["Charging_Access_Score"] + 1))
    train_data["Income_Age_Index"] = (train_data["Annual_Income_USD"] * train_data["Age"])

    categorical_columns = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level"
    ]

    numerical_columns = [
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",

        # engineered features
        "Income_Per_Car",
        "Commute_Environmental_Index",
        "Charging_Access_Score",
        "Commute_Charging_Ratio",
        "Commute_Charging_Index",
        "Income_Age_Index"
    ]

    for col in numerical_columns:
        train_data[col] = train_data[col].fillna(train_data[col].median())
    for col in categorical_columns:
        train_data[col] = train_data[col].fillna(train_data[col].mode()[0])

    train_data = pd.get_dummies(train_data, columns=categorical_columns, drop_first=False,
                                dtype=np.float32)
    train_data[TARGET] = (train_data[TARGET].map({"No": 0, "Yes": 1}).astype(np.float32))

    X = train_data.drop(columns=[TARGET]).values.astype(np.float32)
    y = train_data[TARGET].values.astype(np.float32)

    print(f"Number of features: {X.shape[1]}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = EvPurchasesDataset(X_train, y_train)
    val_dataset = EvPurchasesDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,)

    model = load_model(model_name, with_weights=False, input_dim=X_train.shape[1])
    model.to(device)
    model.train()

    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    pos_weight = num_negative / num_positive

    print(f"Negative samples: {num_negative}")
    print(f"Positive samples: {num_positive}")
    print(f"Positive class weight: {pos_weight}")

    # criterion = nn.BCEWithLogitsLoss(
    #     pos_weight=torch.tensor(
    #         [pos_weight],
    #         dtype=torch.float32,
    #         device=device
    #     )
    # )
    criterion = nn.BCEWithLogitsLoss()

    if weight_decay is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze(1)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()

        val_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch).squeeze(1)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                probs = torch.sigmoid(logits)

                all_probs.extend(probs.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        predictions = (
            all_probs >= 0.5
        ).astype(int)

        accuracy = accuracy_score(all_targets, predictions)
        precision = precision_score(all_targets, predictions, zero_division=0)
        recall = recall_score(all_targets, predictions, zero_division=0)
        f1 = f1_score(all_targets, predictions, zero_division=0)
        auc = roc_auc_score(all_targets, all_probs)

        print(f"Epoch {epoch + 1:02d}/{num_epoch} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Accuracy: {accuracy:.4f} "
        f"| Precision: {precision:.4f} "
        f"| Recall: {recall:.4f} "
        f"| F1: {f1:.4f} "
        f"| AUC: {auc:.4f}")

    save_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2026)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)
    else:
        pass