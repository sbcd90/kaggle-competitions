from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)


class Cifar100Dataset(Dataset):
    def __init__(self, images, labels=None, transform_pipeline: str="train"):
        self.images = images
        self.labels = labels
        self.transform = self.get_transform(transform_pipeline)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image
        return image, int(self.labels[idx])

    def get_transform(self, transform_pipeline):
        if transform_pipeline == "train":
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        elif transform_pipeline == "eval":
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
