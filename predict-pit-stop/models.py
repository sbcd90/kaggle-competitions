from pathlib import Path
import torch
import torch.nn as nn

class PitNextLap(nn.Module):
    def __init__(self, num_features: int=28):
        super(PitNextLap, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2+0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2 + 0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

MODEL_FACTORY = {
    "predict_pit_stop": PitNextLap
}

def load_model(
        model_name: str = "predict_pit_stop",
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