import torch
from torch.utils.data import Dataset
import pandas as pd


class ForecastStickerSalesDataset(Dataset):
    def __init__(self,
                 X_input: pd.DataFrame,
                 y_input: pd.DataFrame):
        if y_input is None:
            self.X = torch.tensor(X_input.values, dtype=torch.float32)
            self.y = None
        else:
            self.X = torch.tensor(X_input.values, dtype=torch.float32)
            self.y = torch.tensor(y_input, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]