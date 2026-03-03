import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
import argparse
import random
from datasets_utils.akkadian_dataset import AkkadianPretrainedDataset, data_collator
from sacrebleu_replacement_api import corpus_bleu, corpus_chrf
import math
import pandas as pd
import os
from peft import PeftModel

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
def test_eval(model, tokenizer, dataloader, device, num_beams=5, max_gen_len=64, print_samples=10):
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

def train(
    model_name: str="translate_akkadian_to_english_pretrained_qlora",
    num_epoch: int=5,
    lr: float=3e-4,
    seed: int=2026,
    batch_size: int=8,
    weight_decay: float=0.01,
    train: bool=True
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU instead")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    src_file = "data/akkadian_output.txt"
    tgt_file = "data/akkadian_output.txt"
    src_lines = open(src_file, "r", encoding="utf-8").read().splitlines()[:1024]
    tgt_lines = open(tgt_file, "r", encoding="utf-8").read().splitlines()[:1024]
    assert len(src_lines) == len(tgt_lines), "Source/Target line counts differ!"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=bnb_config,
                                                 device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q", "v"], lora_dropout=0.05,
                             bias="none", task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = train_val_test_split(src_lines, tgt_lines, seed)

    train_dataset = AkkadianPretrainedDataset(train_src, train_tgt, tokenizer=tokenizer)
    val_dataset = AkkadianPretrainedDataset(val_src, val_tgt, tokenizer=tokenizer)
    collator = data_collator(tokenizer, model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = "qlora_models"
    best_metric = -1.0
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for idx, batch in enumerate(train_loader):
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

def test(
    model_name: str = "translate_akkadian_to_english_pretrained_qlora",
    num_epoch: int = 5,
    lr: float = 3e-4,
    seed: int = 2026,
    batch_size: int = 8,
    weight_decay: float = 0.01,
    train: bool = False
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU instead")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    src_file = "data/akkadian_output.txt"
    tgt_file = "data/akkadian_output.txt"
    src_lines = open(src_file, "r", encoding="utf-8").read().splitlines()[:1024]
    tgt_lines = open(tgt_file, "r", encoding="utf-8").read().splitlines()[:1024]
    assert len(src_lines) == len(tgt_lines), "Source/Target line counts differ!"

    _, _, _, _, test_src, test_tgt = train_val_test_split(src_lines, tgt_lines, seed)

    model_dir = "qlora_models/best"
    base = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    test_dataset = AkkadianPretrainedDataset(test_src, test_tgt, tokenizer=tokenizer)
    collator = data_collator(tokenizer, model)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_eval(model, tokenizer, test_loader, device)

if __name__ == '__main__':
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
        test(**args)