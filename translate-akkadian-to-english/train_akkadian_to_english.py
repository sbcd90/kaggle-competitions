import argparse
import torch
import numpy as np
from torch.utils.data import random_split
import pandas as pd
import math
import sacrebleu
import torch.nn.functional as F

from datasets_utils.akkadian_dataset import AkkadianDataset, collate_fn, PAD_ID, BOS_ID, EOS_ID
from model import load_model, save_model

class WarmupInverseSqrtScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for p in self.optimizer.param_groups:
            p["lr"] = lr

def ids_to_text(ids, tgt_sp):
    ids = [x for x in ids if x not in (PAD_ID, BOS_ID, EOS_ID)]
    return tgt_sp.decode(ids)

def train(
    model_name: str="translate_akkadian_to_english",
    num_epoch: int=50,
    lr: float=1e-2,
    batch_size: int=32,
    seed: int=2026,
    weight_decay: float=None,
    train: bool=True,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    akkadian_dataset = AkkadianDataset(src_file="data/akkadian_output.txt", tgt_file="data/english_output.txt",
        src_spm_path="spm_akk.model", tgt_spm_path="spm_eng.model")

    train_ratio = 0.9
    train_size = int(train_ratio * len(akkadian_dataset))
    val_size = len(akkadian_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(akkadian_dataset,[train_size, val_size],generator=generator)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    src_vocab_size = 4000
    tgt_vocab_size = 8000
    model = load_model(model_name, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    model.to(device)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)#, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)#, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
    scheduler = WarmupInverseSqrtScheduler(optimizer, d_model=256, warmup_steps=2000)

    for epoch in range(num_epoch):
        metrics = {"train_loss": 0.0, "val_loss": 0.0}
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_in)
            optimizer.zero_grad()
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            metrics["train_loss"] += loss.item()
        with torch.inference_mode():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                logits = model(src, tgt_in)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                metrics["val_loss"] += loss.item()
        epoch_train_loss = metrics["train_loss"] / len(train_loader)
        epoch_val_loss = metrics["val_loss"] / len(val_loader)

        # if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        # train_metric = evaluate_bleu_chrf(
        #     model=model,
        #     dataloader=train_loader,
        #     tgt_spm_path="spm_eng.model",
        #     device=device,
        #     max_decode_len=80,
        # )
        # val_metric = evaluate_bleu_chrf(
        #     model=model,
        #     dataloader=val_loader,
        #     tgt_spm_path="spm_eng.model",
        #     device=device,
        #     max_decode_len=80,
        # )
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f}"
            # f"train_bleu={train_metric:.4f}"
            # f"val_bleu={val_metric:.4f}"
        )
    save_model(model)

@torch.no_grad()
def beam_search_decode(
    model,
    src,
    beam_size=5,
    max_len=80,
    length_penalty=0.6
):
    model.eval()
    device = src.device

    beams = [(torch.tensor([[BOS_ID]], device=device), 0.0, False)]

    for _ in range(max_len):
        new_beams = []
        for tokens, score, ended in beams:
            if ended:
                new_beams.append((tokens, score, True))
                continue

            logits = model(src, tokens)
            next_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_logits, dim=-1)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            for k in range(beam_size):
                next_id = topk_ids[0, k].item()
                next_logp = topk_log_probs[0, k].item()

                new_tokens = torch.cat(
                    [tokens, torch.tensor([[next_id]], device=device)],
                    dim=1
                )
                new_score = score + next_logp
                new_ended = (next_id == EOS_ID)
                new_beams.append((new_tokens, new_score, new_ended))

        def normalize_score(tokens, score):
            T = tokens.size(1)
            lp = ((5 + T) / 6) ** length_penalty
            return score / lp

        new_beams.sort(key=lambda x: normalize_score(x[0], x[1]), reverse=True)
        beams = new_beams[:beam_size]

        if all(b[2] for b in beams):
            break
    best_tokens, best_score, _ = beams[0]
    return best_tokens

@torch.no_grad()
def greedy_decode(model, src, max_len=80):
    model.eval()
    device = src.device

    ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits = model(src, ys)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if next_token.item() == EOS_ID:
            break
    return ys

@torch.no_grad()
def evaluate_bleu_chrf(model, dataloader, tgt_spm_path, device, max_decode_len=80):
    import sentencepiece as spm

    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load(tgt_spm_path)

    model.eval()

    refs = []
    hyps = []

    idx = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        for b in range(src.size(0)):
            pred_ids = greedy_decode(model, src[b:b+1], max_len=max_decode_len)[0].tolist()
            gold_ids = tgt[b].tolist()

            pred_text = ids_to_text(pred_ids, tgt_sp)
            gold_text = ids_to_text(gold_ids, tgt_sp)

            hyps.append(pred_text)
            refs.append(gold_text)
            idx += 1

    solution = pd.DataFrame({"id": list(range(len(refs))), "text": refs})
    submission = pd.DataFrame({"id": list(range(len(hyps))), "text": hyps})
    metric_value = score(solution, submission, row_id_column_name="id", text_column_name="text")
    return metric_value

def translate_sentence(sentence: str, model_name: str="translate_akkadian_to_english"):
    import sentencepiece as spm
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    src_sp = spm.SentencePieceProcessor(model_file="spm_akk.model")
    tgt_sp = spm.SentencePieceProcessor(model_file="spm_eng.model")

    model = load_model(model_name, with_weights=True, src_vocab_size=4000,
                       tgt_vocab_size=8000)
    model = model.to(device)
    model.eval()

    src_ids = [BOS_ID] + src_sp.encode(sentence, out_type=int) + [EOS_ID]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    out_ids = greedy_decode(model, src_tensor, max_len=80)[0].tolist()

    out_ids = [x for x in out_ids if x not in (BOS_ID, EOS_ID, PAD_ID)]
    print(tgt_sp.decode(out_ids))

def translate_sentence_beam(sentence: str, model_name: str="translate_akkadian_to_english"):
    import sentencepiece as spm
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    src_sp = spm.SentencePieceProcessor(model_file="spm_akk.model")
    tgt_sp = spm.SentencePieceProcessor(model_file="spm_eng.model")

    model = load_model(model_name, with_weights=True, src_vocab_size=4000,
                       tgt_vocab_size=8000)
    model = model.to(device)
    model.eval()

    src_ids = [BOS_ID] + src_sp.encode(sentence, out_type=int) + [EOS_ID]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    out_ids = beam_search_decode(model, src_tensor, beam_size=5, max_len=80)[0].tolist()
    out_ids = [x for x in out_ids if x not in (BOS_ID, EOS_ID, PAD_ID)]
    print(tgt_sp.decode(out_ids))

class ParticipantVisibleError(Exception):
    pass

def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    text_column_name: str,
) -> float:
    """Calculates the geometric average of BLEU and CHRF++ scores.

    This metric expects the solution and submission dataframes to contain text columns.

    The score is calculated as: sqrt(BLEU * CHRF++)
    Both BLEU and CHRF++ are on a 0-100 scale, so the result will be on a 0-100 scale.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame containing the ground truth text.

    submission : pd.DataFrame
        A DataFrame containing the predicted text.

    row_id_column_name : str
        The name of the column containing the row IDs. This column is removed
        before scoring.

    text_column_name : str
        The name of the column containing the text to be evaluated.

    Returns
    -------
    float
        The geometric mean of the BLEU and CHRF++ scores.

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> text_column_name = "text"
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was not a cat."]
    ... })

    Case: Perfect match
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was not a cat."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    100.0

    Case: Complete mismatch
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["Completely different.", "Nothing alike."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    0.0

    Case: Partial match
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was a cat."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    75.7
    """

    if row_id_column_name in solution.columns:
        del solution[row_id_column_name]
    if row_id_column_name in submission.columns:
        del submission[row_id_column_name]

    # Validate submission column type
    if not (
        pd.api.types.is_string_dtype(submission[text_column_name])
        or pd.api.types.is_object_dtype(submission[text_column_name])
    ):
        raise ParticipantVisibleError(
            f"Submission column '{text_column_name}' must be of string type."
        )

    # Extract lists of strings
    references = solution[text_column_name].astype(str).tolist()
    hypotheses = submission[text_column_name].astype(str).tolist()

    # Calculate BLEU
    # corpus_bleu expects lists of references (list of lists)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    # Calculate CHRF++ (word_order=2)
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    return math.sqrt(bleu.score * chrf.score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)
    else:
        translate_sentence_beam("1 TÚG ša qá-tim i-tur₄-DINGIR il₅-qé", )