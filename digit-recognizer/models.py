import torch
import torch.nn as nn
from pathlib import Path


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return nn.functional.cross_entropy(logits, target)


class DigitRecognizer(nn.Module):

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.cnn_layers(x)
        logits = logits.view(logits.size(0), -1)
        logits = self.fc_layers(logits)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(dim=1)


MODEL_FACTORY = {
    "digit_recognizer": DigitRecognizer
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
