import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms.v2 import RandomRotation


class DigitRecognizerDataset(Dataset):
    def __init__(self,
                 x_input: pd.DataFrame,
                 y_input: pd.Series,
                 transform_pipeline: str = "default"):
        self.transform = self.get_transform(transform_pipeline)

        if y_input is None:
            self.X = x_input.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = None
        else:
            self.X = x_input.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(y_input.values)
        print()


    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if self.y is not None:
            return self.transform(self.X[item]), self.y[item]
        else:
            return self.transform(self.X[item])

    def get_transform(self, transform_pipeline: str="default"):
        xform = None

        if transform_pipeline == "default":
            xform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                ]
            )
        elif transform_pipeline == "aug":
            xform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    RandomRotation(degrees=20),
                    #RandomAffine(degrees=0, translate=(0.2, 0.2), resample=Image.Resampling.BICUBIC, fill=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                ]
            )

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")
        return xform

