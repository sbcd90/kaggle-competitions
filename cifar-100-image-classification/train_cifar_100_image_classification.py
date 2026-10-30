import argparse
import pickle
import pandas as pd

import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

from datasets.cifar_100_dataset import Cifar100Dataset
from model import load_model, save_model


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='bytes')

def train(
        model_name: str="cifar_100",
        num_epochs: int=50,
        lr: float=0.001,
        batch_size: int=64,
        seed: int=2026,
        weight_decay: float=0.0005,
        train: bool=True,
        **kwargs
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dict = load_pickle("data/cifar-100-image-classification-challenge/train")
    meta_dict = load_pickle("data/cifar-100-image-classification-challenge/meta")

    X = train_dict[b'data']
    y = np.array(train_dict[b"fine_labels"], dtype=np.int64)

    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X = np.ascontiguousarray(X)

    class_names = [name.decode("utf-8") for name in meta_dict[b'fine_label_names']]

    print(f"Training images: {X.shape}")
    print(f"Training labels: {y.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Example classes: {class_names[:10]}")

    train_full = Cifar100Dataset(X, labels=y, transform_pipeline="train")
    val_full = Cifar100Dataset(X, labels=y, transform_pipeline="eval")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(X), generator=generator).tolist()

    num_val = int(len(X) * 0.1)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = load_model(model_name, **kwargs)
    model = model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,)

    for epoch in range(num_epochs):
        model.train()
        scheduler.step()
        metrics = {"train_acc": [], "val_acc": []}

        for img, label in train_loader:
            img = img.to(device)
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
                img = img.to(device)
                label = label.to(device)

                out = model(img)
                _, predicted = torch.max(out, 1)
                matched = torch.sum(torch.eq(label, predicted)).item()
                metrics["val_acc"].append(matched / batch_size)

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        print(
            f"Epoch {epoch + 1:2d} / {num_epochs:2d}: "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_acc={epoch_val_acc:.4f}"
        )
    save_model(model)

def test(
        model_name: str="cifar_100",
        batch_size: int=64,
        train: bool=False,
        **kwargs
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = load_model(model_name, with_weights=True, **kwargs)
    model = model.to(device)
    model.eval()

    test_dict = load_pickle("data/cifar-100-image-classification-challenge/test")
    X_test = test_dict[b"data"]
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    X_test = np.ascontiguousarray(X_test)

    test_dataset = Cifar100Dataset(X_test, labels=None, transform_pipeline="eval")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,)

    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            logits = model(images)
            predicted_labels = logits.argmax(dim=1)

            predictions.extend(predicted_labels.cpu().tolist())

    submission = pd.DataFrame({"id": range(len(predictions)), "label": predictions,})
    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Submission rows:", len(submission))
    print("Saved submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)
    else:
        test(args["model_name"], args["batch_size"], args["train"])