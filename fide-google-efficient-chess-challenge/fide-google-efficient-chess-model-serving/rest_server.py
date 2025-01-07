from flask import Flask, request
import torch.nn as nn
import torch
from pathlib import Path
import os
import chess
import numpy as np

app = Flask(__name__)

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
    "fide_google_chess_model_moved_from": MoveModel,
    "fide_google_chess_model_moved_to": MoveModel
}

def load_model(
        model_name: str = "fide_google_chess_model",
        with_weights: bool = True,
        **model_kwargs,
) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = Path(os.path.abspath('')).resolve() / f"fide-google-efficient-chess-model-serving/{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    return m

def board_position_list_one_hot(board):
    one_hot = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x = square // 8
        y = square % 8
        piece_type = piece.piece_type - 1
        color_offset = 6 if piece.color == chess.BLACK else 0
        one_hot[x, y, piece_type + color_offset] = 1
    return one_hot

@app.route('/', methods=['POST'])
def serve_chess_moves():
    fen = request.data.decode('utf-8')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    board_position_one_hot_list = board_position_list_one_hot(chess.Board(fen)).tolist()
    board_position_list_input = torch.tensor(board_position_one_hot_list).permute(2, 0, 1).unsqueeze(0).to(device)

    kwargs = {}

    model_moved_from = load_model(model_name="fide_google_chess_model_moved_from", with_weights=True, **kwargs)
    model_moved_from = model_moved_from.to(device)
    model_moved_from.eval()

    y_pred = torch.nn.functional.softmax(model_moved_from(board_position_list_input))
    _, move_from = torch.max(y_pred, 1)

    model_moved_to = load_model(model_name="fide_google_chess_model_moved_to", with_weights=True, **kwargs)
    model_moved_to = model_moved_to.to(device)
    model_moved_to.eval()

    y_pred = torch.nn.functional.softmax(model_moved_to(board_position_list_input))
    _, moved_to = torch.max(y_pred, 1)
    piece_moved_pair = {"moved_from": move_from.item(), "moved_to": moved_to.item()}
    return piece_moved_pair


if __name__ == '__main__':
    app.run()