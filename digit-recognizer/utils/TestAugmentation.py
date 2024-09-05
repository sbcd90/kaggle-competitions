import pandas as pd
from av.deprecation import method
from torchvision.transforms.v2 import RandomRotation
from torchvision import transforms

from matplotlib import pyplot as plt
import numpy as np
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

rotate = RandomRotation(20)
shift = RandomShift(3)
composed = transforms.Compose(
    [
        RandomRotation(20),
        RandomShift(3)
    ]
)

fig = plt.figure()
train_df = pd.read_csv("../data/train.csv")
sample = transforms.ToPILImage()(train_df.iloc[65, 1:].values.reshape((28,28)).astype(np.uint8)[:,:,None])

for i, transform in enumerate([rotate, shift, composed]):
    transformed_sample = transform(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(transform).__name__)
    ax.imshow(np.reshape(np.array(list(transformed_sample.getdata())), (-1, 28)), cmap='gray')
plt.show()
print()