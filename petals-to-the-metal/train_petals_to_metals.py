import glob

import pandas as pd
import torch
import numpy as np
import tensorflow as tf
import argparse
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets.petals_to_metals_dataset import PetalsToMetalsDataset
from models import load_model, save_model, ClassificationLoss

feature_description = {
    "class": tf.io.FixedLenFeature([], tf.int64),
    "id": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string)
}

feature_description_test = {
    "id": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string)
}

def __parse_image_function(example_photo):
    return tf.io.parse_single_example(example_photo, feature_description)

def __parse_image_function_test(example_photo):
    return tf.io.parse_single_example(example_photo, feature_description_test)

def train(
        model_name: str = "petals_to_metal",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2024,
        weight_decay: bool = False,
        train: bool = True,
        **kwargs,
) -> nn.Module:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_files = glob.glob("data/tpu-getting-started/*/train/*.tfrec")
    val_files = glob.glob("data/tpu-getting-started/*/val/*.tfrec")

    train_ids = []
    train_classes = []
    train_images = []

    for i in train_files:
        train_image_dataset = tf.data.TFRecordDataset(i)
        train_image_dataset = train_image_dataset.map(__parse_image_function)

        ids = [str(id_features["id"].numpy())[2: -1] for id_features in train_image_dataset]
        train_ids.extend(ids)

        classes = [int(class_features["class"].numpy()) for class_features in train_image_dataset]
        train_classes.extend(classes)

        images = [image_features["image"].numpy() for image_features in train_image_dataset]
        train_images.extend(images)

    val_ids = []
    val_classes = []
    val_images = []

    for i in val_files:
        val_image_dataset = tf.data.TFRecordDataset(i)
        val_image_dataset = val_image_dataset.map(__parse_image_function)

        ids = [str(id_features["id"].numpy())[2: -1] for id_features in val_image_dataset]
        val_ids.extend(ids)

        classes = [int(class_features["class"].numpy()) for class_features in val_image_dataset]
        val_classes.extend(classes)

        images = [image_features["image"].numpy() for image_features in val_image_dataset]
        val_images.extend(images)

#    sample = transforms.ToPILImage()(ToTensor()((Image.open(io.BytesIO(val_images[1])))))
#    plt.imshow(sample)
#    plt.axis("off")
#    plt.show()

    train_dataset = PetalsToMetalsDataset(train_ids, train_classes, train_images, transform_pipeline="aug")
    val_dataset = PetalsToMetalsDataset(val_ids, val_classes, val_images)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, with_weights=False, **kwargs)
    model = model.to(device)

    loss_func = ClassificationLoss()
    if weight_decay is True:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        model.train()
        metrics = {"train_acc": [], "val_acc": []}

        for train_data in train_loader:
            img = train_data[0].to(device)
            label = train_data[1].to(device)

            out = model(img)

            optimizer.zero_grad()
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out, 1)
            matched = torch.sum(torch.eq(label, predicted)).item()
            metrics["train_acc"].append(matched / batch_size)

        with torch.inference_mode():
            for val_data in val_loader:
                img = val_data[0].to(device)
                label = val_data[1].to(device)

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
        break
    save_model(model)
    return model

def test(
        trained_model: nn.Module,
        batch_size: int = 32
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    test_files = glob.glob("data/tpu-getting-started/*/test/*.tfrec")
    test_ids = []
    test_classes = []
    test_images = []

    for i in test_files:
        test_image_dataset = tf.data.TFRecordDataset(i)
        test_image_dataset = test_image_dataset.map(__parse_image_function_test)

        ids = [str(id_features["id"].numpy())[2: -1] for id_features in test_image_dataset]
        test_ids.extend(ids)

        images = [image_features["image"].numpy() for image_features in test_image_dataset]
        test_images.extend(images)

    test_dataset = PetalsToMetalsDataset(test_ids, test_classes, test_images)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    image_ids = []
    predictions = []
    trained_model = trained_model.to(device)
    trained_model.eval()

    with torch.inference_mode():
        for test_data in test_loader:
            img = test_data[0].to(device)
            ids = test_data[2]
            image_ids.extend(list(ids))

            out = torch.nn.functional.softmax(trained_model(img))
            _, predicted = torch.max(out, 1)
            predictions.extend(predicted.cpu().numpy())
    submission = pd.DataFrame({"id": np.arange(1, len(predictions) + 1), "label": predictions})
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
    parser.add_argument("--weight_decay", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        # pass all arguments to train
        model = train(**args)
        test(model, args["batch_size"])
    else:
        model = load_model(model_name=args["model_name"])
        test(model, args["batch_size"])
