import pickle
import torch
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from forecast_datasets.forecasting_sticker_sales_dataset import ForecastStickerSalesDataset
from models import load_model, save_model


def transform_datetime(df: pd.DataFrame):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 30)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 30)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def get_holidays():
    return {
        'Kenya': pd.date_range(start='2010-01-01', end='2030-12-31', freq='B').append(
            pd.to_datetime(['2023-04-07', '2023-04-09', '2023-05-01', '2023-06-01', '2023-10-10', '2023-12-12'])),
        'Canada': pd.date_range(start='2010-01-01', end='2030-12-31', freq='B').append(pd.to_datetime(
            ['2023-02-20', '2023-04-14', '2023-05-22', '2023-07-01', '2023-09-04', '2023-10-09', '2023-12-25'])),
        'Finland': pd.date_range(start='2010-01-01', end='2030-12-31', freq='B').append(
            pd.to_datetime(['2023-01-06', '2023-04-14', '2023-05-01', '2023-06-24', '2023-12-06', '2023-12-25'])),
        'Norway': pd.date_range(start='2010-01-01', end='2030-12-31', freq='B').append(
            pd.to_datetime(['2023-05-01', '2023-05-17', '2023-12-25', '2023-12-26'])),
        'Singapore': pd.date_range(start='2010-01-01', end='2030-12-31', freq='B').append(
            pd.to_datetime(['2023-01-22', '2023-04-07', '2023-06-05', '2023-08-09', '2023-11-12', '2023-12-25'])),
        'Italy': pd.date_range(start='2010-01-01', end='2030-12-31', freq='B').append(pd.to_datetime(
            ['2023-01-06', '2023-04-25', '2023-05-01', '2023-06-02', '2023-08-15', '2023-11-01', '2023-12-08',
             '2023-12-25']))
    }

def check_holiday(row):
    holidays = get_holidays()
    country = row['country']
    date = row['date']
    return 1 if date in holidays[country].values else 0

def train(
        model_name: str = "forecasting_sticker_sales",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2025,
        weight_decay: float = None,
        train: bool = True
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = pd.read_csv("data/train.csv")
    train_data["num_sold"] = train_data["num_sold"].fillna(0)
    train_data = transform_datetime(train_data)
    train_data["season"] = train_data["month"].apply(get_season)
    # train_data["is_holiday"] = train_data.apply(check_holiday, axis=1)

    categorical_cols = ["country", "store", "product", "season"]
    label_encoders = {col: LabelEncoder() for col in categorical_cols}

    for col in categorical_cols:
        train_data[col] = label_encoders[col].fit_transform(train_data[col].astype(str))
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    numerical_cols = ["month", "year", "day", "day_of_week", "week_of_year", "quarter", "day_sin", "day_cos", "month_sin", "month_cos"]
    scaler = StandardScaler()

    train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

    X_train = train_data.drop(columns=["id", "num_sold", "date"])
    y_train = train_data["num_sold"].values

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    train_dataset = ForecastStickerSalesDataset(X_train, y_train)
    val_dataset = ForecastStickerSalesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, num_features=X_train.shape[1])
    model = model.to(device)
    model.train()

    loss_func = torch.nn.MSELoss()
    if weight_decay is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    expr_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=50)

    for epoch in range(num_epoch):
        metrics = {"train_loss": 0.0, "val_loss": 0.0}
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)

            loss = loss_func(y_pred, y)
            metrics["train_loss"] += loss.item() * len(X)

            loss.backward()
            optimizer.step()

        with torch.inference_mode():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                loss = loss_func(y_pred, y)
                metrics["val_loss"] += loss.item() * len(X)

        metrics["train_loss"] /= len(train_loader.dataset)
        metrics["val_loss"] /= len(val_loader.dataset)

        epoch_train_rmse_loss = torch.sqrt(torch.as_tensor(metrics["train_loss"] / len(train_loader.dataset)))
        epoch_val_rmse_loss = torch.sqrt(torch.as_tensor(metrics["val_loss"] / len(val_loader.dataset)))
        expr_lr_scheduler.step(metrics=epoch_val_rmse_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"Learning Rate: {current_lr:.6f} "
            f"train_loss={epoch_train_rmse_loss:.4f} "
            f"val_loss={epoch_val_rmse_loss:.4f}"
        )

    save_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        # pass all arguments to train
        train(**args)
    else:
        pass
        #test(args["model_name"], args["batch_size"])
