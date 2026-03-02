import torch
import numpy as np
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
import argparse


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
    max_memory = {
        0: "7.5GiB",
        1: "7.5GiB",
        "cpu": "40GiB"
    }
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=bnb_config,
                                                 device_map="auto", max_memory=max_memory)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, force_download=True)
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q", "v"], lora_dropout=0.1,
                             bias="none", task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # model.to(device)
    pass

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
        pass