import argparse

import numpy as np
import pandas as pd
import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models import ClassificationLoss, load_model, save_model


def train(
        model_name: str = "digit_recognizer",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
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

    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    x = train_data.drop(["label"], axis=1).values
    y = train_data["label"].values

    x_test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    test_dataset = TensorDataset(x_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    loss_func = ClassificationLoss()
    if weight_decay is True:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        metrics = {"train_acc": [], "val_acc": []}

        for img, label in train_loader:
            img = img.to(device).view(-1, 1, 28, 28)
            label = label.to(device)

            out = model(img)

            optimizer.zero_grad()
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out, 1)
            matched = torch.sum(torch.eq(label, predicted)).item()
            metrics["train_acc"].append(matched / batch_size)

        with torch.inference_mode():
            for img, label in val_loader:
                img = img.to(device).view(-1, 1, 28, 28)
                label = label.to(device)

                out = torch.nn.functional.softmax(model(img))
                _, predicted = torch.max(out, 1)
                matched = torch.sum(torch.eq(label, predicted)).item()
                metrics["val_acc"].append(matched / batch_size)

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=bool, default=False)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
