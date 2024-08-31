import argparse

import numpy as np
import pandas as pd
import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def train(
        model_name: str = "digit_recognizer",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 128,
        seed: int = 2024,
        weight_decay: bool = False,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    x = train.drop(["label"], axis=1).values
    y = train["label"].values

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    x_test_tensor = torch.tensor(test.values, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for epoch in range(num_epoch):
        for img, label in train_loader:
            img = img.to(device).view(-1, 1, 28, 28)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=bool, default=False)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
