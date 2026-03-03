import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HF_TOKEN = "****"

login(token=HF_TOKEN)

model_name = "akk-to-eng-mt5-small"
out_dir = "models"

model_dir = os.path.join(out_dir, "best")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

repo_name = f"sbcd90/{model_name}"

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
print("Model successfully pushed to Hugging Face Hub!")