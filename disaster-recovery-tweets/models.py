import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched = False):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x: torch.Tensor):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        emb_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, emb_size))).type(torch.LongTensor).to(device)
        if self.batched is True:
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int,
                 num_positions = 20, dropout = 0.1, output_dim = 2):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=True)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout,
                                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.output_dim = output_dim

    def forward(self, indices: torch.Tensor):
        embedded_indices = self.char_embedding(indices)
        embedded_indices = self.positional_encoding(embedded_indices)
        transformer_output = self.transformer_encoder(embedded_indices)
        logits = self.output_layer(transformer_output)
        logits = logits.mean(dim=1)
        probs = nn.functional.softmax(logits, dim=-1)
        return probs

MODEL_FACTORY = {
    "disaster_recovery_tweets": TransformerEncoderModel
}

def load_model(
        model_name: str,
        with_weights: bool = False,
        **model_kwargs,
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


def save_model(
        model: nn.Module
) -> str:
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = Path(__file__).resolve().parent / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path
