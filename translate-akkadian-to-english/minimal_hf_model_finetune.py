import math
import pandas as pd
import sacrebleu
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

# -------------------------
# Your Kaggle metric
# -------------------------
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, text_column_name: str) -> float:
    if row_id_column_name in solution.columns:
        del solution[row_id_column_name]
    if row_id_column_name in submission.columns:
        del submission[row_id_column_name]

    references = solution[text_column_name].astype(str).tolist()
    hypotheses = submission[text_column_name].astype(str).tolist()

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    return math.sqrt(bleu.score * chrf.score)


# -------------------------
# Tiny dataset
# -------------------------
class MyDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=128):
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
        )

        labels = self.tokenizer(
            text_target=tgt,
            max_length=self.max_len,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# -------------------------
# Beam eval
# -------------------------
@torch.no_grad()
def evaluate(model, tokenizer, dataloader, device, num_beams=5, max_gen_len=128):
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

        pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # IMPORTANT: labels contain -100 from collator -> fix before decoding
        #labels = batch["labels"].clone()
        #labels[labels == -100] = tokenizer.pad_token_id
        gold_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        hyps.extend([x.strip() for x in pred_texts])
        refs.extend([x.strip() for x in gold_texts])

    solution = pd.DataFrame({"id": list(range(len(refs))), "text": refs})
    submission = pd.DataFrame({"id": list(range(len(hyps))), "text": hyps})

    metric_value = score(solution, submission, "id", "text")
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score

    return metric_value, bleu, chrf


# -------------------------
# Main
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    # Example data (replace with your Akkadian pairs)
    src_texts = [
        "translate: šarrum dannum",
        "translate: bītum rabûm",
        "translate: awīlum idû",
        "translate: šarrum awīlum",
        "translate: bītum dannum",
    ]
    tgt_texts = [
        "a strong king",
        "a big house",
        "the man knows",
        "the king is a man",
        "a strong house",
    ]

    # train/val split
    split = int(0.8 * len(src_texts))
    train_src, val_src = src_texts[:split], src_texts[split:]
    train_tgt, val_tgt = tgt_texts[:split], tgt_texts[split:]

    train_ds = MyDataset(train_src, train_tgt, tokenizer)
    val_ds = MyDataset(val_src, val_tgt, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=3e-4)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0

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

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"VAL Metric sqrt(BLEU*CHRF): {metric_value:.2f}")
        print(f"VAL BLEU: {bleu:.2f} | VAL CHRF++: {chrf:.2f}")


if __name__ == "__main__":
    main()
