from torch.utils.data import Dataset
import pandas as pd
import torch

class UsedCarPricesDataset(Dataset):
    def __init__(self,
                 X_input: pd.DataFrame,
                 y_input: pd.DataFrame):
        if y_input is None:
            self.X = torch.tensor(X_input, dtype=torch.float32)
            self.y = None
        else:
            self.X = torch.tensor(X_input, dtype=torch.float32)
            self.y = torch.tensor(y_input, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if self.y is not None:
            return self.X[item], self.y[item]
        else:
            return self.X[item]
