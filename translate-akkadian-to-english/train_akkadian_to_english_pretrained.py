import math
import argparse
import os
import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from sacrebleu_replacement_api import corpus_bleu, corpus_chrf

from datasets_utils.akkadian_dataset import AkkadianPretrainedDataset, data_collator

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, text_column_name: str) -> float:
    if row_id_column_name in solution.columns:
        del solution[row_id_column_name]
    if row_id_column_name in submission.columns:
        del submission[row_id_column_name]

    references = solution[text_column_name].astype(str).tolist()
    hypotheses = submission[text_column_name].astype(str).tolist()

    bleu = corpus_bleu(hypotheses, [references])
    chrf = corpus_chrf(hypotheses, [references], word_order=2)

    return math.sqrt(bleu.score * chrf.score)

@torch.no_grad()
def evaluate(model, tokenizer, dataloader, device, num_beams=5, max_gen_len=64):
    model.eval()

    hyps, refs = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_beams=num_beams,
            max_length=max_gen_len,
            early_stopping=True,
        )

        try:
            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        except OverflowError as e:
            print("OverflowError during prediction decoding:", e)
            pred_texts = [""] * generated.size(0)

        # IMPORTANT: labels contain -100 from collator -> fix before decoding
        #labels = batch["labels"].clone()
        #labels[labels == -100] = tokenizer.pad_token_id
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        try:
            gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except OverflowError as e:
            print("OverflowError during gold decoding:", e)
            gold_texts = [""] * labels.size(0)

        hyps.extend([x.strip() for x in pred_texts])
        refs.extend([x.strip() for x in gold_texts])

    solution = pd.DataFrame({"id": list(range(len(refs))), "text": refs})
    submission = pd.DataFrame({"id": list(range(len(hyps))), "text": hyps})

    metric_value = score(solution, submission, "id", "text")
    bleu = corpus_bleu(hyps, [refs]).score
    chrf = corpus_chrf(hyps, [refs], word_order=2).score

    return metric_value, bleu, chrf

@torch.no_grad()
def test(model, tokenizer, dataloader, device, num_beams=5, max_gen_len=64, print_samples=10):
    model.eval()
    hyps, refs = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_beams=num_beams,
            max_length=max_gen_len,
            early_stopping=True,
        )

        try:
            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        except OverflowError as e:
            print("OverflowError during prediction decoding:", e)
            pred_texts = [""] * generated.size(0)

        # IMPORTANT: labels contain -100 from collator -> fix before decoding
        # labels = batch["labels"].clone()
        # labels[labels == -100] = tokenizer.pad_token_id
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        try:
            gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except OverflowError as e:
            print("OverflowError during gold decoding:", e)
            gold_texts = [""] * labels.size(0)

        hyps.extend([x.strip() for x in pred_texts])
        refs.extend([x.strip() for x in gold_texts])

    print("\n=== Sample Predictions ===\n")
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        if i >= print_samples:
            break

        print(f"Example {i + 1}")
        print(f"REF: {ref}")
        print(f"HYP: {hyp}")
        print("-" * 50)

def train_val_test_split(src_lines, tgt_lines, seed):
    assert len(src_lines) == len(tgt_lines), "Source/Target line counts differ!"

    data = list(zip(src_lines, tgt_lines))
    random.seed(seed)
    random.shuffle(data)

    src_lines, tgt_lines = zip(*data)

    n = len(src_lines)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_src = src_lines[:train_end]
    train_tgt = tgt_lines[:train_end]

    val_src = src_lines[train_end:val_end]
    val_tgt = tgt_lines[train_end:val_end]

    test_src = src_lines[val_end:]
    test_tgt = tgt_lines[val_end:]

    return (
        train_src, train_tgt, val_src, val_tgt, test_src, test_tgt,
    )

def train(
    model_name: str="translate_akkadian_to_english_pretrained",
    num_epoch: int=5,
    lr: float=5e-4,
    seed: int=2026,
    batch_size: int=8,
    weight_decay: float=0.01,
    train: bool = True,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    src_file = "data/akkadian_output.txt"
    tgt_file = "data/english_output.txt"
    src_lines = open(src_file, "r", encoding="utf-8").read().splitlines()[:128]
    tgt_lines = open(tgt_file, "r", encoding="utf-8").read().splitlines()[:128]
    assert len(src_lines) == len(tgt_lines), "Source/Target line counts differ!"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = train_val_test_split(src_lines, tgt_lines, seed)

    train_dataset = AkkadianPretrainedDataset(train_src, train_tgt, tokenizer=tokenizer)
    val_dataset = AkkadianPretrainedDataset(val_src, val_tgt, tokenizer=tokenizer)
    collator = data_collator(tokenizer, model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = "models"
    best_metric = -1.0
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        metric_value, bleu, chrf = evaluate(model, tokenizer, val_loader, device)

        print(f"\nEpoch {epoch + 1}/{num_epoch}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"VAL Metric sqrt(BLEU*CHRF): {metric_value:.2f}")
        print(f"VAL BLEU: {bleu:.2f} | VAL CHRF++: {chrf:.2f}")

        if metric_value > best_metric:
            best_metric = metric_value
            save_path = os.path.join(out_dir, "best")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved new best model to: {save_path} (Metric={best_metric:.2f})")

    print("Start testing....")
    model_dir = os.path.join(out_dir, "best")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    test_dataset = AkkadianPretrainedDataset(test_src, test_tgt, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test(model, tokenizer, test_loader, device)

# class ParticipantVisibleError(Exception):
#     pass
#
# def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, text_column_name: str) -> float:
#     if row_id_column_name in solution.columns:
#         del solution[row_id_column_name]
#     if row_id_column_name in submission.columns:
#         del submission[row_id_column_name]
#
#     if submission[text_column_name].dtype not in ("object", "string"):
#         raise ParticipantVisibleError(
#             f"Submission column '{text_column_name}' must be of string type."
#         )
#
#     references = solution[text_column_name].astype(str).tolist()
#     hypotheses = submission[text_column_name].astype(str).tolist()
#
#     bleu = sacrebleu.corpus_bleu(hypotheses, [references])
#     chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
#
#     return math.sqrt(bleu.score * chrf.score)
#
# def set_seed(seed=42):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#
# def load_parallel(src_file, tgt_file):
#     src_lines = open(src_file, "r", encoding="utf-8").read().splitlines()
#     tgt_lines = open(tgt_file, "r", encoding="utf-8").read().splitlines()
#     assert len(src_lines) == len(tgt_lines), "Source/Target line counts differ!"
#     return src_lines, tgt_lines
#
# def make_dataset(src_lines, tgt_lines):
#     return Dataset.from_dict({
#         "src": src_lines,
#         "tgt": tgt_lines,
#     })
#
# def train_val_split(dataset, val_ratio=0.1, seed=42):
#     dataset = dataset.shuffle(seed=seed)
#     n = len(dataset)
#     n_val = int(n * val_ratio)
#     val_ds = dataset.select(range(n_val))
#     train_ds = dataset.select(range(n_val, n))
#     return train_ds, val_ds
#
# def preprocess_function(batch, tokenizer, max_src_len=128, max_tgt_len=128):
#     model_inputs = tokenizer(
#         batch["src"],
#         max_length=max_src_len,
#         truncation=True,
#         padding=False
#     )
#     labels = tokenizer(
#         batch["tgt"],
#         max_length=max_tgt_len,
#         truncation=True,
#         padding=False
#     )
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# @torch.no_grad()
# def evaluate(model, tokenizer, dataloader, device, num_beams=5, max_gen_len=128):
#     model.eval()
#
#     hyps = []
#     refs = []
#
#     for batch in dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#
#         generated = model.generate(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             num_beams=num_beams,
#             max_length=max_gen_len,
#             early_stopping=True,
#         )
#
#         pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
#         gold_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
#
#         hyps.extend([x.strip() for x in pred_texts])
#         refs.extend([x.strip() for x in gold_texts])
#
#     solution = pd.DataFrame({"id": list(range(len(refs))), "text": refs})
#     submission = pd.DataFrame({"id": list(range(len(hyps))), "text": hyps})
#
#     metric_value = score(solution, submission, "id", "text")
#
#     bleu  = sacrebleu.corpus_bleu(hyps, [refs]).score
#     chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
#
#     return metric_value, bleu, chrf
#
# def main():
#     set_seed(2026)
#
#     model_name = "google/mt5-small"
#
#     src_file = "data/akkadian_output.txt"
#     tgt_file = "data/english_output.txt"
#
#     out_dir = "mt5_akk_en_ckpt"
#     os.makedirs(out_dir, exist_ok=True)
#
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         print("CUDA is not available. Using CPU instead.")
#         device = torch.device("cpu")
#
#     src_lines, tgt_lines = load_parallel(src_file, tgt_file)
#     dataset = make_dataset(src_lines, tgt_lines)
#     train_ds, val_ds = train_val_split(dataset, val_ratio=0.1, seed=2026)
#
#     print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
#
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     model.to(device)
#
#     max_src_len = 128
#     max_tgt_len = 128
#
#     train_tok = train_ds.map(
#         lambda x: preprocess_function(x, tokenizer, max_src_len, max_tgt_len),
#         batched=True,
#         remove_columns=("src", "tgt"),
#     )
#     val_tok = val_ds.map(
#         lambda x: preprocess_function(x, tokenizer, max_src_len, max_tgt_len),
#         batched=True,
#         remove_columns=("src", "tgt"),
#     )
#
#     data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
#
#     train_loader = DataLoader(train_tok, batch_size=8, shuffle=True, collate_fn=data_collator)
#     val_loader = DataLoader(val_tok, batch_size=8, shuffle=False, collate_fn=data_collator)
#
#     lr = 5e-4
#     weight_decay = 0.01
#     epochs = 5
#     grad_clip = 1.0
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     total_steps = epochs * len(train_loader)
#     warmup_steps = int(0.1 * total_steps)
#
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
#                                                 num_training_steps=total_steps)
#
#     use_amp = torch.cuda.is_available()
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#
#     best_metric = -1.0
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#
#         for step, batch in enumerate(train_loader):
#             batch = {k: v.to(device) for k, v in batch.items()}
#
#             outputs = model(**batch)
#             loss = outputs.loss
#
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
#
#             total_loss += loss.item()
#
#             if (step + 1) % 200 == 0:
#                 print(f"Epoch {epoch} Step {step + 1}/{len(train_loader)} Loss {loss.item():.4f}")
#             break
#         avg_loss = total_loss / len(train_loader)
#
#         metric, bleu, chrf = evaluate(model=model, tokenizer=tokenizer, dataloader=val_loader,
#             device=device, num_beams=5, max_gen_len=128
#         )
#
#         print(
#             f"\nEpoch {epoch:02d} | TrainLoss={avg_loss:.4f} | "
#             f"Metric={metric:.2f} | BLEU={bleu:.2f} | CHRF++={chrf:.2f}\n"
#         )
#
#         if metric > best_metric:
#             best_metric = metric
#             save_path = os.path.join(out_dir, "best")
#             model.save_pretrained(save_path)
#             tokenizer.save_pretrained(save_path)
#             print(f"✅ Saved new best model to: {save_path} (Metric={best_metric:.2f})")
#
#     print("Training done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)
    else:
        pass