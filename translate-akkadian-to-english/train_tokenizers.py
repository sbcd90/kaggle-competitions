import sentencepiece as spm

def train_spm(input_file, model_prefix, vocab_size=8000):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=1+2
    )

train_spm("data/akkadian_output.txt", "spm_akk", vocab_size=4000)
train_spm("data/english_output.txt", "spm_eng", vocab_size=8000)