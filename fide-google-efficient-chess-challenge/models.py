import torch.nn as nn
import torch
from pathlib import Path

class MoveModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=1024, num_classes=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=128, kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 64)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

MODEL_FACTORY = {
    "fide_google_chess_model": MoveModel
}

def load_model(
        model_name: str = "fide_google_chess_model",
        with_weights: bool = True,
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