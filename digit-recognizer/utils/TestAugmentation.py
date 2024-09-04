import pandas as pd
from torchvision.transforms.v2 import RandomRotation
from torchvision import transforms

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, img):
        # Apply the random affine transformation without resampling
        affine_transform = transforms.RandomAffine(degrees=0, translate=(self.shift, self.shift))
        img = affine_transform(img)

        # Apply resampling separately using PIL
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, 0), resample=Image.BICUBIC)

        return img

rotate = RandomRotation(20)
shift = RandomShift(0.2)
composed = transforms.Compose(
    [
        RandomRotation(20),
        RandomShift(0.2)
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