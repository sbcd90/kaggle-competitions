from torch import dtype
from torch.utils.data import Dataset
import torch

class FideGoogleChessDataset(Dataset):
    def __init__(self, X_input, y_input):
        self.X = torch.tensor(X_input, dtype=torch.float32)
        self.y = torch.tensor(y_input, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx], None
        return self.X[idx].permute(2, 0, 1), self.y[idx]