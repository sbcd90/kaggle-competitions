from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "sbcd90/akk-to-eng-mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print()