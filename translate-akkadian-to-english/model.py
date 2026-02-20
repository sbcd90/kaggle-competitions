import math
import torch
import torch.nn as nn
from pathlib import Path

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerAkToEngModel(nn.Module):
     def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=512,
        pad_id=0
     ):
         super().__init__()
         self.pad_id = pad_id
         self.d_model = d_model

         self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=self.pad_id)
         self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=self.pad_id)

         self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
         self.dropout = nn.Dropout(p=dropout)

         self.transformer = nn.Transformer(
             d_model=d_model,
             nhead=nhead,
             num_encoder_layers=num_encoder_layers,
             num_decoder_layers=num_decoder_layers,
             dim_feedforward=dim_feedforward,
             dropout=dropout,
             batch_first=True,
         )
         self.fc_out = nn.Linear(d_model, tgt_vocab_size)

     def make_src_key_padding_mask(self, src):
         return (src == self.pad_id)

     def make_tgt_key_padding_mask(self, tgt):
         return (tgt == self.pad_id)

     def make_tgt_causal_mask(self, tgt_len, device):
         mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
         return mask

     def forward(self, src, tgt_in):
         device = src.device
         B, S = src.shape
         B, T = tgt_in.shape

         src_key_padding_mask = self.make_src_key_padding_mask(src)
         tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt_in)
         tgt_causal_mask = self.make_tgt_causal_mask(T, device)

         src_emb = self.dropout(self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model)))
         tgt_emb = self.dropout(self.pos_enc(self.tgt_emb(tgt_in) * math.sqrt(self.d_model)))

         out = self.transformer(
             src=src_emb,
             tgt=tgt_emb,
             src_key_padding_mask=src_key_padding_mask,
             tgt_key_padding_mask=tgt_key_padding_mask,
             memory_key_padding_mask=src_key_padding_mask,
             tgt_mask=tgt_causal_mask,
         )

         logits = self.fc_out(out)
         return logits

MODEL_FACTORY = {
    "translate_akkadian_to_english": TransformerAkToEngModel
}

def load_model(
        model_name: str = "predict_intro_extroverts",
        with_weights:bool = False,
        **model_kwargs
) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    return m

def save_model(model: nn.Module) -> str:
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = Path(__file__).resolve().parent / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path