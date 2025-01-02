import numpy as np
import torch
import h5py
import argparse
import os

from sklearn.model_selection import train_test_split
from chess_datasets.fide_google_chess_dataset import FideGoogleChessDataset
from torch.utils.data import DataLoader
from models import load_model, save_model
from torch.optim import lr_scheduler

# python3 train_fide_google_chess.py --model_name=fide_google_chess_model --lr=0.0001 --weight_decay=True
def train(
        model_name: str = "fide_google_chess_model",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2025,
        weight_decay: bool = False,
        is_moved_from: bool = False,
        **kwargs
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_input = None
    moved_from = None
    moved_to = None

    for file in os.listdir("training_data"):
        # load training data
        with h5py.File(f"training_data/{file}", "r") as f:
            if train_input is None:
                train_input = torch.tensor(f["input_position"][:])
                moved_from = torch.tensor(f["moved_from"][:])
                moved_to = torch.tensor(f["moved_to"][:])
            else:
                train_input = torch.cat((train_input, torch.tensor(f["input_position"][:])), dim=0)
                moved_from = torch.cat((moved_from, torch.tensor(f["moved_from"][:])), dim=0)
                moved_to = torch.cat((moved_to, torch.tensor(f["moved_to"][:])), dim=0)

    train_input_ndarray = train_input.numpy()
    moved_from_ndarray = moved_from.numpy()
    moved_to_ndarray = moved_to.numpy()

    if is_moved_from:
        moved_ndarray = moved_from_ndarray
    else:
        moved_ndarray = moved_to_ndarray

    X_train, X_test, y_train, y_test = train_test_split(train_input_ndarray, moved_ndarray, test_size=0.2, random_state=seed)

    train_dataset = FideGoogleChessDataset(X_train, y_train)
    val_dataset = FideGoogleChessDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, with_weights=False, **kwargs)
    model = model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    if weight_decay is True:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    expr_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=50)

    for epoch in range(num_epoch):
        model.train()
        metrics = {"train_acc": [], "val_acc": []}

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_pred, 1)
            matched = torch.sum(torch.eq(predicted, y)).item()
            metrics["train_acc"].append(matched / batch_size)

        with torch.inference_mode():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                y_pred = torch.nn.functional.softmax(model(X))
                _, predicted = torch.max(y_pred, 1)
                matched = torch.sum(torch.eq(predicted, y)).item()
                metrics["val_acc"].append(matched / batch_size)

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        expr_lr_scheduler.step(metrics=epoch_val_acc)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
    save_model(model)

def test(
        model_name: str = "fide_google_chess_model",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2025,
        weight_decay: bool = False,
        is_moved_from: bool = True,
        **kwargs
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(model_name, with_weights=True, **kwargs)
    model = model.to(device)
    model.eval()

    train_input = None
    moved_from = None
    moved_to = None

    for file in os.listdir("training_data"):
        # load training data
        with h5py.File(f"training_data/{file}", "r") as f:
            if train_input is None:
                train_input = torch.tensor(f["input_position"][:])
                moved_from = torch.tensor(f["moved_from"][:])
                moved_to = torch.tensor(f["moved_to"][:])
            else:
                train_input = torch.cat((train_input, torch.tensor(f["input_position"][:])), dim=0)
                moved_from = torch.cat((moved_from, torch.tensor(f["moved_from"][:])), dim=0)
                moved_to = torch.cat((moved_to, torch.tensor(f["moved_to"][:])), dim=0)

    train_input_ndarray = train_input.numpy()[:2]
    moved_from_ndarray = moved_from.numpy()[:2]
    moved_to_ndarray = moved_to.numpy()[:2]

    if is_moved_from:
        moved_ndarray = moved_from_ndarray
    else:
        moved_ndarray = moved_to_ndarray
    test_dataset = FideGoogleChessDataset(train_input_ndarray, moved_ndarray)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = torch.nn.functional.softmax(model(X))
            _, predicted = torch.max(y_pred, 1)
            matched = torch.sum(torch.eq(predicted, y)).item()
            print(f"Matched: {matched}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=bool, default=False)
    parser.add_argument("--is_moved_from", type=bool, default=False)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
    #test(**vars(parser.parse_args()))