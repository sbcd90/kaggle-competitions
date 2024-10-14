import torch
import torch.nn as nn
from pathlib import Path

class Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.LayerNorm(output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, output_channels),
            nn.LayerNorm(output_channels),
            nn.ReLU()
        )

        if input_channels != output_channels:
            self.skip = nn.Linear(input_channels, output_channels)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        y = self.model(x)
        return y + self.skip(x)

class UsedCarPricesModel(nn.Module):
    def __init__(self, input_channels: int = 11):
        super().__init__()

#        layer_size = 128
#        num_layers = 4
#        c = input_channels

#        layers = nn.ModuleList()
#        layers.append(nn.Linear(c, layer_size, bias=False))
#        c = layer_size

#        for _ in range(num_layers):
#            layers.append(Block(c, layer_size))
#            c = layer_size
#        layers.append(nn.Linear(c, 1))
#        self.model = nn.Sequential(*layers)
        self.fc1 = nn.Linear(input_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return self.model(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

model_factory = {
    "used_car_prices": UsedCarPricesModel
}

def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

def save_model(model):
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")

def load_model(model_name: str, with_weights: bool=True, **model_kwargs):
    """
        Called by the grader to load a pre-trained model by name
        """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r