import torch
import torch.nn as nn
from pathlib import Path

class IntroExtrovertModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 51),
            nn.Dropout(0.336682495915063),
            nn.ReLU(),
            nn.Linear(51, 16),
            nn.Dropout(0.2899754225673039),
            nn.ReLU(),
            nn.Linear(16, 80),
            nn.Dropout(0.4363500829739193),
            nn.ReLU(),
            nn.Linear(80, 48),
            nn.Dropout(0.37355415962243),
            nn.Linear(48, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

MODEL_FACTORY = {
    "predict_intro_extroverts": IntroExtrovertModel
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
