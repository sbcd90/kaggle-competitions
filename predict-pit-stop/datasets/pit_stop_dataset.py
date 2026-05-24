import torch
from torch.utils.data import Dataset

class PitStopDataset(Dataset):
    def __init__(self, X, y):
        if y is None:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = None
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]