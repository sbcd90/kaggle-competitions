import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from intro_extrovert_datasets.intro_extrovert_dataset import IntroExtrovertDataset
from models import load_model, save_model

class Preprocessing:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def impute_categorical(self):
        for col in self.test.select_dtypes(include=["object"]).columns:
            self.train[col].fillna(self.train[col].mode()[0], inplace=True)
            self.test[col].fillna(self.test[col].mode()[0], inplace=True)

    def impute_numerical(self):
        for col in self.test.select_dtypes(include=["float64", "int64"]).columns:
            self.train[col].fillna(self.train[col].mean(), inplace=True)
            self.test[col].fillna(self.test[col].mean(), inplace=True)

    def encode_categorical(self):
        mapper = {"Yes": 0, "No": 1}
        for col in self.test.select_dtypes(include=["object"]).columns:
            self.train[col] = self.train[col].map(mapper)
            self.test[col] = self.test[col].map(mapper)

    def encode_target(self):
        mapper = {"Introvert": 0, "Extrovert": 1}
        self.train["Personality"] = self.train["Personality"].map(mapper)

    def preprocess(self):
        self.impute_categorical()
        self.impute_numerical()
        self.encode_categorical()
        self.encode_target()
        return self.train, self.test

def train(
        model_name: str = "predict_intro_extroverts",
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

    train_data = pd.read_csv("data/playground-series-s5e7/train.csv")
    test_data = pd.read_csv("data/playground-series-s5e7/test.csv")

    preprocessor = Preprocessing(train_data, test_data)
    preprocessor.preprocess()
    train_data, _ = preprocessor.train, preprocessor.test

    X_train = train_data.drop(columns=["id", "Personality"])
    y_train = train_data["Personality"].values

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    train_dataset = IntroExtrovertDataset(X_train, y_train)
    val_dataset = IntroExtrovertDataset(X_val, y_val)

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

    global_step = 0
    for epoch in range(num_epoch):
        metrics = {"train_acc": [], "val_acc": []}
        for data, labels in train_loader:
            data, labels = data.to(device), torch.tensor(labels, dtype=torch.long).to(device)
            out = model(data)

            optimizer.zero_grad()
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out, 1)
            matched = torch.sum(torch.eq(predicted, labels))
            metrics["train_acc"].append(matched.item() / len(labels))

            global_step += 1
        with torch.inference_mode():
            for data, labels in val_loader:
                data, labels = data.to(device), torch.tensor(labels, dtype=torch.long).to(device)
                out = model(data)

                _, predicted = torch.max(out, 1)
                matched = torch.sum(torch.eq(predicted, labels))
                metrics["val_acc"].append(matched.item() / len(labels))

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
        model_name: str = "predict_intro_extroverts",
        seed: int = 2025,
        train: bool = False,
        **kwargs
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = pd.read_csv("data/playground-series-s5e7/train.csv")
    test_data = pd.read_csv("data/playground-series-s5e7/test.csv")

    preprocessor = Preprocessing(train_data, test_data)
    preprocessor.preprocess()
    _, test_data = preprocessor.train, preprocessor.test

    X_test = test_data.drop(columns=["id"])
    y_test = test_data["id"].values

    test_dataset = IntroExtrovertDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = load_model(model_name, with_weights=True, num_features=X_test.shape[1])
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
            labels = ["Introvert" if p == 0 else "Extrovert" for p in predicted.cpu()]
            test_predictions.extend(labels)
    df = pd.DataFrame({"id": test_ids, "Personality": test_predictions})
    df.to_csv("submission.csv", index=False)


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
        train(**args)
    else:
        test(**args)