import argparse
import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from datasets.pit_stop_dataset import PitStopDataset
from torch.utils.data import DataLoader
from models import load_model, save_model


def train(
        model_name: str = "predict_pit_stop",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2026,
        weight_decay: float = 1e-5,
        train: bool = True,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = pd.read_csv("data/train.csv")
    train_df = train_data.sort_values(["Driver", "Race", "LapNumber"]).reset_index(drop=True)

    train_df["WearPerLap"] = (train_df["Cumulative_Degradation"] / (train_df["TyreLife"] + 1))
    train_df["RemainingRace"] = (1.0 - train_df["RaceProgress"])
    train_df["StintPressure"] = (train_df["TyreLife"] * train_df["Position"])
    train_df["PaceDrop"] = (train_df["LapTime_Delta"] / (np.abs(train_df["Cumulative_Degradation"]) + 1))
    train_df["TrafficRisk"] = (train_df["Position"] * train_df["LapTime_Delta"])

    def get_race_phase(x):
        if x < 0.33:
            return 0
        elif x < 0.66:
            return 1
        else:
            return 2
    train_df["RacePhase"] = (train_df["RaceProgress"].apply(get_race_phase))

    average_stint = (train_df.groupby("Driver")["TyreLife"].mean().to_dict())
    train_df["DriverAverageTyreLife"] = (train_df["Driver"].map(average_stint))

    compound_degradation = (train_df.groupby("Compound")["Cumulative_Degradation"].mean().to_dict())
    train_df["CompoundAverageDegradation"] = (train_df["Compound"].map(compound_degradation))

    train_df["DegradationTrend"] = (train_df.groupby(["Driver", "Race"])["Cumulative_Degradation"].diff().fillna(0))
    train_df["LapDeltaTrend"] = (train_df.groupby(["Driver", "Race"])["LapTime_Delta"].diff().fillna(0))

    train_df["RollingDegradation"] = (train_df.groupby(["Driver", "Race"])["Cumulative_Degradation"]
                                      .rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    train_df["RollingPaceLoss"] = (train_df.groupby(["Driver", "Race"])["LapTime_Delta"]
                                      .rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    train_df["TyreLife2"] = (train_df["TyreLife"] ** 2)
    train_df["PitWindowsPressure"] = (train_df["TyreLife"] * train_df["WearPerLap"])
    train_df["PositionTrend"] = (train_df.groupby(["Driver", "Race"])["Position"].diff().fillna(0))
    train_df = train_df.fillna(0)

    categorical_cols = ["Driver", "Compound", "Race"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        encoders[col] = le
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    feature_cols = ["Driver", "Compound", "Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
                    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position_Change", "WearPerLap", "RemainingRace",
                    "StintPressure", "PaceDrop", "TrafficRisk", "RacePhase", "DriverAverageTyreLife",
                    "CompoundAverageDegradation", "DegradationTrend", "LapDeltaTrend", "RollingDegradation", "RollingPaceLoss",
                    "TyreLife2", "PitWindowsPressure", "PositionTrend"]
    X = train_df[feature_cols]
    y = train_df["PitNextLap"]

    splitter = GroupShuffleSplit(test_size=0.2, random_state=seed)
    train_idx, valid_idx = next(splitter.split(train_df, groups=train_df["Race"]))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_dataset = PitStopDataset(X_train, y_train)
    valid_dataset = PitStopDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, with_weights=False)
    model.to(device)
    model.train()

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count

    positive_weight = (negative_count / positive_count)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight],
                                                             dtype=torch.float32))
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
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            predicted = (out >= 0.5).int()
            matched = torch.sum(torch.eq(predicted, labels))
            metrics["train_acc"].append((matched.item() / len(labels)))
            metrics["train_loss"] += loss.item()
        with torch.inference_mode():
            for data, labels in valid_loader:
                data, labels = data.to(device), labels.to(device)
                out = model(data)

                loss = criterion(out, labels)
                predicted = (out >= 0.5).int()
                matched = torch.sum(torch.eq(predicted, labels))
                metrics["val_acc"].append((matched.item() / len(labels)))
                metrics["val_loss"] += loss.item()

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        epoch_train_loss = metrics["train_loss"] / len(train_loader)
        epoch_val_loss = metrics["val_loss"] / len(valid_loader)

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_acc={epoch_val_acc:.4f} "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f} "
        )
    save_model(model)

def test_pit_stop(
    model_name: str = "predict_pit_stop",
    num_epoch: int = 50,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 2026,
    weight_decay: float = 1e-5,
    train: bool = False,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU instead")
        device = torch.device("cpu")

    test_data = pd.read_csv("data/test.csv")
    test_df = test_data.sort_values(["Driver", "Race", "LapNumber"]).reset_index(drop=True)

    test_df["WearPerLap"] = (test_df["Cumulative_Degradation"] / (test_df["TyreLife"] + 1))
    test_df["RemainingRace"] = (1.0 - test_df["RaceProgress"])
    test_df["StintPressure"] = (test_df["TyreLife"] * test_df["Position"])
    test_df["PaceDrop"] = (test_df["LapTime_Delta"] / (np.abs(test_df["Cumulative_Degradation"]) + 1))
    test_df["TrafficRisk"] = (test_df["Position"] * test_df["LapTime_Delta"])

    def get_race_phase(x):
        if x < 0.33:
            return 0
        elif x < 0.66:
            return 1
        else:
            return 2

    test_df["RacePhase"] = (test_df["RaceProgress"].apply(get_race_phase))

    average_stint = (test_df.groupby("Driver")["TyreLife"].mean().to_dict())
    test_df["DriverAverageTyreLife"] = (test_df["Driver"].map(average_stint))

    compound_degradation = (test_df.groupby("Compound")["Cumulative_Degradation"].mean().to_dict())
    test_df["CompoundAverageDegradation"] = (test_df["Compound"].map(compound_degradation))

    test_df["DegradationTrend"] = (test_df.groupby(["Driver", "Race"])["Cumulative_Degradation"].diff().fillna(0))
    test_df["LapDeltaTrend"] = (test_df.groupby(["Driver", "Race"])["LapTime_Delta"].diff().fillna(0))

    test_df["RollingDegradation"] = (test_df.groupby(["Driver", "Race"])["Cumulative_Degradation"]
                                      .rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    test_df["RollingPaceLoss"] = (test_df.groupby(["Driver", "Race"])["LapTime_Delta"]
                                   .rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    test_df["TyreLife2"] = (test_df["TyreLife"] ** 2)
    test_df["PitWindowsPressure"] = (test_df["TyreLife"] * test_df["WearPerLap"])
    test_df["PositionTrend"] = (test_df.groupby(["Driver", "Race"])["Position"].diff().fillna(0))
    test_df = test_df.fillna(0)

    categorical_cols = ["Driver", "Compound", "Race"]
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    for col in categorical_cols:
        le = encoders[col]
        test_df[col] = le.transform(test_df[col])

    feature_cols = ["Driver", "Compound", "Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
                    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position_Change", "WearPerLap",
                    "RemainingRace",
                    "StintPressure", "PaceDrop", "TrafficRisk", "RacePhase", "DriverAverageTyreLife",
                    "CompoundAverageDegradation", "DegradationTrend", "LapDeltaTrend", "RollingDegradation",
                    "RollingPaceLoss",
                    "TyreLife2", "PitWindowsPressure", "PositionTrend"]
    X = test_df[feature_cols]

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X = scaler.transform(X)
    y = test_df["id"]

    test_dataset = PitStopDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = load_model(model_name, with_weights=True)
    model.to(device)
    model.eval()

    test_ids = []
    test_likelihoods = []
    with torch.inference_mode():
        for data, ids in test_loader:
            data = data.to(device)
            out = torch.sigmoid(model(data))

            test_ids.extend(torch.tensor(ids, dtype=torch.long).tolist())
            labels = [p for p in out.tolist()]
            test_likelihoods.extend(labels)
    df = pd.DataFrame({"id": test_ids, "PitNextLap": test_likelihoods})
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
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
        test_pit_stop(**args)




