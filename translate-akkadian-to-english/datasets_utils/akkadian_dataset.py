from typing import List

import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from transformers import DataCollatorForSeq2Seq

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

class AkkadianDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_spm_path, tgt_spm_path, max_len=128):
        self.src_lines = open(src_file, "r", encoding="utf-8").read().splitlines()
        self.tgt_lines = open(tgt_file, "r", encoding="utf-8").read().splitlines()
        assert len(self.src_lines) == len(self.tgt_lines)

        self.src_sp = spm.SentencePieceProcessor()
        self.src_sp.load(src_spm_path)

        self.tgt_sp = spm.SentencePieceProcessor()
        self.tgt_sp.load(tgt_spm_path)

        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def encode(self, sp, text):
        ids = sp.encode(text, out_type=int)
        ids = [BOS_ID] + ids + [EOS_ID]
        return ids[:self.max_len]

    def __getitem__(self, idx):
        src = self.encode(self.src_sp, self.src_lines[idx])
        tgt = self.encode(self.tgt_sp, self.tgt_lines[idx])
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]

    src_max = max(src_lens)
    tgt_max = max(tgt_lens)

    src_padded = torch.full((len(batch), src_max), PAD_ID, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max), PAD_ID, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    return src_padded, tgt_padded

class AkkadianPretrainedDataset(Dataset):
    def __init__(self, src_texts: List[str], tgt_texts: List[str], tokenizer, max_len=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.src_texts[idx]
        tgt = self.tgt_texts[idx]

        model_inputs = self.tokenizer(
            src,
            max_length=self.max_len,
            truncation=True,
            padding=False
        )
        labels = self.tokenizer(
            text_target=tgt,
            max_length=self.max_len,
            truncation=True,
            padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def data_collator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    return data_collator