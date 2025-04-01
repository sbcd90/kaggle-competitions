import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

from datasets.gi_tract_dataset import GiTractDataset
from models import load_model, save_model


def train(
        model_name: str = "uw_mgi_segment_model",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2025,
        weight_decay: float = None,
        is_train: bool = True,
        **kwargs
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("cuda is not available, using cpu")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    DATA_DIR = "data/uw-madison-gi-tract-image-segmentation"
    input_df = pd.read_csv(f"{DATA_DIR}/train.csv")

    train_df, val_df = train_test_split(input_df, test_size=0.2, random_state=seed)
    train_df["class_id"] = pd.Categorical(train_df["class"]).codes
    val_df["class_id"] = pd.Categorical(val_df["class"]).codes

    train_dataset = GiTractDataset(train_df, DATA_DIR)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = GiTractDataset(val_df, DATA_DIR)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, with_weights=False)
    model = model.to(device)
    model.train()

    loss_func = torch.nn.CrossEntropyLoss()
    if weight_decay is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print("No weight decay")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=50)

    for epoch in range(num_epoch):
        metrics = {"training_loss": 0.0, "val_loss": 0.0}

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            out = model(images)
            optimizer.zero_grad()

            loss = loss_func(out, masks.long())
            loss.backward()
            optimizer.step()

            metrics["training_loss"] += loss.item()

        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                out = model(images)
                loss = loss_func(out, masks.long())
                metrics["val_loss"] += loss.item()
        print(f"Epoch {epoch}: Training loss: {metrics['training_loss'] / len(train_loader)}, "
              f"Validation loss: {metrics['val_loss'] / len(val_loader)}")
        exp_lr_scheduler.step(metrics["val_loss"] / len(val_loader))
    save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="uw_mgi_segment_model")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)