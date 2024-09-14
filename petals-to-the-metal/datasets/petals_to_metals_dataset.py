import io

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PetalsToMetalsDataset(Dataset):
    def __init__(self,
                 ids,
                 classes,
                 images,
                 transform_pipeline: str = "default",
                 data_category: str = "train"):
        self.transform = self.get_transform(transform_pipeline)

        self.ids = ids
        self.classes = classes
        self.images = images
        self.data_category = data_category

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        image = self.images[item]
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        if self.data_category != "test":
            return image, int(self.classes[item]), self.ids[item]
        else:
            return image, 0, self.ids[item]

    def get_transform(self, transform_pipeline: str = "default"):
        xform = None

        if transform_pipeline == "default":
            xform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif transform_pipeline == "aug":
            xform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        elif transform_pipeline == "example":
            xform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")
        return xform