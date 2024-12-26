import argparse
import os
import site

from models import load_model, save_model

os.environ["SP_DIR"] = site.getsitepackages()[0]

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from dataset.disaster_recovery_datasets import DisasterRecoveryDataset, collate_fn


# python3 train_disaster_recovery_tweets.py --model_name=disaster_recovery_tweets --lr=0.0004 --weight_decay=True --num_epoch=140
def __yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)

def train(
        model_name: str = "disaster_recovery_tweets",
        num_epoch: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        seed: int = 2024,
        weight_decay: bool = False,
        train: bool = True,
        **kwargs
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = pd.read_csv("data/nlp-getting-started/train.csv")

    sentences = list(train_data["text"])
    labels = list(train_data["target"])
    tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(__yield_tokens(sentences, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=seed
    )

    train_dataset = DisasterRecoveryDataset(train_sentences, train_labels, tokenizer, vocab)
    val_dataset = DisasterRecoveryDataset(val_sentences, val_labels, tokenizer, vocab)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(vocab)
    num_layers = 2
    num_heads = 1
    num_positions = 128
    d_model = 128

    model = load_model(model_name, with_weights=False, vocab_size=vocab_size, d_model=d_model,
                       num_layers=num_layers, num_heads=num_heads, num_positions=num_positions)
    model = model.to(device)
    model.train()

    loss_func = torch.nn.CrossEntropyLoss()
    if weight_decay is True:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(num_epoch):
        metrics = {"train_acc": [], "val_acc": []}

        for sentences, labels in train_data:
            sentences, labels = sentences.to(device), torch.tensor(labels).to(device)

            out = model(sentences)

            optimizer.zero_grad()
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out, 1)
            matched = torch.sum(torch.eq(labels, predicted))
            metrics["train_acc"].append(matched / batch_size)

            global_step += 1
        with torch.inference_mode():
            for sentences, labels in val_data:
                sentences, labels = sentences.to(device), torch.tensor(labels).to(device)

                out = model(sentences)

                _, predicted = torch.max(out, 1)
                matched = torch.sum(torch.eq(labels, predicted))
                metrics["val_acc"].append(matched / batch_size)

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
    save_model(model)

def test(
        model_name: str = "disaster_recovery_tweets",
        train: bool = False,
        **kwargs
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(2024)
    np.random.seed(2024)

    train_sentences = pd.read_csv("data/nlp-getting-started/train.csv")
    train_sentences = list(train_sentences["text"])
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(__yield_tokens(train_sentences, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    test_data = pd.read_csv("data/nlp-getting-started/test.csv")
    test_sentences = list(test_data["text"])
    test_ids = list(test_data["id"])

    vocab_size = len(vocab)
    num_layers = 2
    num_heads = 1
    num_positions = 128
    d_model = 128

    model = load_model(model_name, with_weights=True, vocab_size=vocab_size, d_model=d_model,
                       num_layers=num_layers, num_heads=num_heads, num_positions=num_positions)
    model = model.to(device)
    model.eval()

    test_dataset = DisasterRecoveryDataset(test_sentences, test_ids, tokenizer, vocab)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    test_ids = []
    test_predictions = []
    with torch.inference_mode():
        for sentences, ids in test_data:
            sentences = sentences.to(device)
            out = model(sentences)
            _, predicted = torch.max(out, 1)

            test_ids.extend(ids)
            test_predictions.extend(predicted.cpu().numpy())

    df = pd.DataFrame({"id": test_ids, "target": test_predictions})
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=bool, default=False)

    # pass all arguments to train
    #train(**vars(parser.parse_args()))
    test(**vars(parser.parse_args()))
