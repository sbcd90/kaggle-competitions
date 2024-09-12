import torch
import torch.nn as nn
from pathlib import Path

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return nn.functional.cross_entropy(logits, target)

class PetalsToMetals(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            layers = []
            c1 = in_channels

            n_layers = 1
            for _ in range(n_layers):
                layers.append(nn.Conv2d(c1, out_channels, kernel_size, stride, padding))
                layers.append(nn.ReLU())
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding))
                layers.append(nn.ReLU())
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding))
                layers.append(nn.ReLU())
                c1 = out_channels
            self.model = nn.Sequential(*layers)

            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            return self.model(x) + self.skip(x)

    def __init__(self, in_channels=3, num_classes=104):
        super().__init__()

        channels_l0 = 64
        cnn_layers = [
            nn.Conv2d(in_channels, channels_l0, kernel_size=11, stride=2, padding=(11 - 1) // 2),
            nn.ReLU()
        ]

        c1 = channels_l0
        n_blocks = 1
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2

        cnn_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        cnn_layers.append(nn.Dropout(p=0.5))
        cnn_layers.append(nn.Conv2d(c1, num_classes, kernel_size=1))
        cnn_layers.append(nn.Flatten())
        self.network = nn.Sequential(*cnn_layers)

    def forward(self, x):
        return self.network(x)

MODEL_FACTORY = {
    "petals_to_metal": PetalsToMetals
}

def load_model(
        model_name: str,
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