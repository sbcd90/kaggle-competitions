import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd
from typing import Callable, List

from torchtext.vocab import Vocab


class DisasterRecoveryDataset(Dataset):
    def __init__(self,
                 x_input,
                 y_input,
                 tokenizer: Callable[[str], List[str]],
                 vocab: Vocab):
        self.x_input = x_input
        self.y_input = y_input
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.x_input)

    def __getitem__(self, item):
        tokens = self.tokenizer(self.x_input[item])
        word_to_indices = self.vocab(tokens)
        if self.y_input is not None:
            return torch.tensor(word_to_indices), self.y_input[item]
        else:
            return torch.tensor(word_to_indices)

def collate_fn(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    padded_batch = pad_sequence(features, batch_first=True, padding_value=0)
    return padded_batch, labels