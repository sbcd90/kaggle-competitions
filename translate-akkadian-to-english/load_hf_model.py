from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", force_download=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
print()