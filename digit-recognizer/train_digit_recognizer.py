import argparse

import numpy as np
import pandas as pd
import torch.cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models import load_model, save_model
from datasets.digit_recognizer_dataset import DigitRecognizerDataset


# python3 train_digit_recognizer.py --model_name=digit_recognizer --weight_decay=True --lr=0.0001 --num_epoch=200
def train(
        model_name: str = "digit_recognizer",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2024,
        weight_decay: bool = False,
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

    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    test_result = pd.read_csv("data/submission.csv")

    x_train = train_data.drop(["label"], axis=1)
    y_train = train_data["label"]
    x_val = test_data
    y_val = test_result["Label"]

    #x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    train_dataset = DigitRecognizerDataset(x_train, y_train, transform_pipeline="aug")
    val_dataset = DigitRecognizerDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, **kwargs)
    model = model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    if weight_decay is True:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epoch):
        model.train()
        exp_lr_scheduler.step()
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


def test(
        model_name: str = "digit_recognizer",
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

    test_data = pd.read_csv("data/test.csv")
    test_dataset = DigitRecognizerDataset(test_data, None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = []

    with torch.inference_mode():
        for img in test_loader:
            img = img.to(device).view(-1, 1, 28, 28)
            out = torch.nn.functional.softmax(model(img))

            _, predicted = torch.max(out, 1)
            predictions.extend(predicted.cpu().numpy())
    submission = pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions})
    submission.to_csv("mnist_classify.csv", index=False)


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
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        # pass all arguments to train
        train(** args)
    else:
        test(args["model_name"], args["batch_size"], args["train"])
