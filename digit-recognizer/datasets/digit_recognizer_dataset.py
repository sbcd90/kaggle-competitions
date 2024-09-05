import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms.v2 import RandomRotation
from PIL import Image

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift

    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift

    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)

        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)


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
                    RandomShift(shift=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                ]
            )

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")
        return xform

