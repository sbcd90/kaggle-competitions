import argparse
from cProfile import label

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models import load_model, save_model
import pickle

from datasets.regression_of_used_car_prices_dataset import UsedCarPricesDataset

# python3 train_regression_of_used_car_prices.py --model_name=used_car_prices --lr=0.0002 --weight_decay=0.001 --batch_size=64 --num_epoch=64 --seed=42 --train=True
def train(
        model_name: str = "used_car_prices",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2024,
        weight_decay: float = None,
        train: bool = True,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = pd.read_csv("data/playground-series-s4e9/train.csv")

    categorical_cols = ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col",
                        "accident", "clean_title"]
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    for col in categorical_cols:
        train_data[col] = label_encoders[col].fit_transform(train_data[col].astype(str))

    numerical_cols = ["model_year", "milage"]
    scaler = StandardScaler()

    train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

    X_train = train_data.drop(columns=["id", "price"])
    y_train = train_data["price"].values

    X_train, X_val, y_train, y_val = train_test_split(X_train.values, y_train, test_size=0.08, random_state=seed)

    train_dataset = UsedCarPricesDataset(X_train, y_train)
    val_dataset = UsedCarPricesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    loss_func = torch.nn.MSELoss()
    if weight_decay is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print("weight_decay is None")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=50)

    for epoch in range(num_epoch):
        metrics = {"training_loss": 0.0, "val_loss": 0.0}

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            out = model(X)

            optimizer.zero_grad()

            loss = loss_func(out.squeeze(), y)
            metrics["training_loss"] += loss.item() * X.shape[0]

            loss.backward()
            optimizer.step()

        with torch.inference_mode():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                out = model(X)

                loss = loss_func(out.squeeze(), y)
                metrics["val_loss"] += loss.item() * X.shape[0]

        epoch_train_rmse_loss = torch.sqrt(torch.as_tensor(metrics["training_loss"] / len(train_loader.dataset)))
        epoch_val_rmse_loss = torch.sqrt(torch.as_tensor(metrics["val_loss"] / len(val_loader.dataset)))
        exp_lr_scheduler.step(metrics=epoch_val_rmse_loss)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"Learning Rate: {current_lr:.6f} "
                f"train_loss={epoch_train_rmse_loss:.4f} "
                f"val_loss={epoch_val_rmse_loss:.4f}"
            )
    save_model(model)

def test(
        model_name: str = "used_car_prices",
        batch_size: int = 32,
        train: bool = False,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.eval()

    test_data = pd.read_csv("data/playground-series-s4e9/test.csv")

    categorical_cols = ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col",
                        "accident", "clean_title"]
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    for col in categorical_cols:
        test_data[col] = test_data[col].map(lambda s: label_encoders[col].classes_.tolist().index(s)
            if s in label_encoders[col].classes_ else -1)

    numerical_cols = ["model_year", "milage"]
    scaler = StandardScaler()

    test_data[numerical_cols] = scaler.fit_transform(test_data[numerical_cols])

    X_test = test_data.drop(columns=["id"])
    y_test = test_data["id"].values
    test_dataset = UsedCarPricesDataset(X_test.values, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ids = []
    predictions = []
    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(device)
            ids.extend(y.tolist())

            out = model(X)
            predictions.extend(out.cpu().numpy().tolist())
    ids = list(map(lambda x: int(x), ids))
    predictions = list(map(lambda x: x[0], predictions))
    submission = pd.DataFrame({"id": ids, "price": predictions})
    submission.to_csv("submission.csv", index=False)


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
        train(** args)
    else:
        test(args["model_name"], args["batch_size"])