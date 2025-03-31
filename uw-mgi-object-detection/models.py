import torch
import torch.nn as nn
from pathlib import Path

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.MaxPool2d(2)
        )

        self.bottleneck = DoubleConv(512, 1024)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[2](self.encoder[1](enc1))
        enc4 = self.encoder[4](self.encoder[3](enc2))
        enc5 = self.encoder[6](self.encoder[5](enc4))

        bottleneck = self.bottleneck(self.encoder[7](enc5))

        dec1 = self.decoder[1](torch.cat([self.decoder[0](bottleneck), enc5], dim=1))
        dec2 = self.decoder[3](torch.cat([self.decoder[2](dec1), enc4], dim=1))
        dec4 = self.decoder[5](torch.cat([self.decoder[4](dec2), enc2], dim=1))
        dec5 = self.decoder[7](torch.cat([self.decoder[6](dec4), enc1], dim=1))

        return self.final_conv(dec5)

model_factory = {
    "uw_mgi_segment_model": UNet
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

def load_model(model_name: str = "uw_mgi_segment_model", with_weights: bool=True):
    """
            Called by the grader to load a pre-trained model by name
            """
    r = model_factory[model_name]()
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    return r
