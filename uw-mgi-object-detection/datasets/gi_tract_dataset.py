import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import glob

def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)
    if not mask_rle or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)
    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]
    ends = [start + length for start, length in zip(starts, lengths)]

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(shape)

def get_full_file_name(prefix: str) -> str:
    files = glob.glob(prefix)
    return files[0] if files else None

class GiTractDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform if transform is not None else A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["id"]
        mask_rle = self.df.iloc[idx]["segmentation"]
        class_id = self.df.iloc[idx]["class_id"]

        directory = img_id.split("_")[0]
        sub_directory = img_id.split("_")[0] + "_" + img_id.split("_")[1]
        scan = img_id.split("_")[2] + "_" + img_id.split("_")[3]

        img_path_prefix = f"{self.data_dir}/train/{directory}/{sub_directory}/scans/{scan}*"
        img_path = get_full_file_name(img_path_prefix)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())

        mask = rle_decode(mask_rle, image.shape)
        mask = mask * class_id

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.long()