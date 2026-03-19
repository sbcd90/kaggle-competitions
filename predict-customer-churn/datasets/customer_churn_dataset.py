import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomerChurnDataset(Dataset):
    def __init__(self, X_input: pd.DataFrame, y_input: pd.DataFrame=None):
        if y_input is None:
            self.X = torch.tensor(X_input.values, dtype=torch.float32)
            self.y = None
        else:
            self.X = torch.tensor(X_input.values, dtype=torch.float32)
            self.y = torch.tensor(y_input.values, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index]
        else:
            return self.X[index], self.y[index]