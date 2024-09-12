from operator import itemgetter

import torch
from tensorflow.python.ops.metrics_impl import mean_per_class_accuracy
from torch.utils.data import DataLoader
from datasets.petals_to_metals_dataset import PetalsToMetalsDataset
import glob
import tensorflow as tf
from train_petals_to_metals import __parse_image_function
import matplotlib.pyplot as plt


train_files = glob.glob("../data/tpu-getting-started/*/train/*.tfrec")
val_files = glob.glob("../data/tpu-getting-started/*/val/*.tfrec")

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

example_set = PetalsToMetalsDataset(train_ids, train_classes, train_images, transform_pipeline="example")
x, y, _ = next(iter(DataLoader(example_set)))

channels = ["Red", "Green", "Blue"]
cmaps = [plt.cm.Reds_r, plt.cm.Greens_r, plt.cm.Blues_r]
fig, ax = plt.subplots(1, 4, figsize=(15, 10))

for i, axs in enumerate(fig.axes[:3]):
    axs.imshow(x[0][i, :, :], cmap=cmaps[i])
    axs.set_title(f'{channels[i]} Channel')
    axs.set_xticks([])
    axs.set_yticks([])

ax[3].imshow(x[0].permute(1,2,0))
ax[3].set_title('Three Channels')
ax[3].set_xticks([])
ax[3].set_yticks([])

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

train_dataset = PetalsToMetalsDataset(train_ids, train_classes, train_images, transform_pipeline="aug")
val_dataset = PetalsToMetalsDataset(val_ids, val_classes, val_images)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

dataset_sizes = {
    "train": len(train_dataset),
    "val": len(val_dataset)
}

loaders = {
    "train": train_loader,
    "val": val_loader
}

for channel in range(len(channels)):
    for x in ["train", "val"]:
        num_pxl = dataset_sizes[x] * 224 * 224

        total_sum = 0
        for batch in loaders[x]:
            layer = list(map(itemgetter(channel), batch[0]))
            layer = torch.stack(layer, dim=0)
            total_sum += layer.sum()
        mean = total_sum / num_pxl

        sum_sqrt = 0
        for batch in loaders[x]:
            layer = list(map(itemgetter(channel), batch[0]))
            sum_sqrt += ((torch.stack(layer, dim=0) - mean).pow(2)).sum()
        std = torch.sqrt(sum_sqrt / num_pxl)

        print(f"|channel:{channel+1}| {x} - mean: {mean}, std: {std}")