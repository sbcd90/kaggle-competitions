from Chessnut import Game
import numpy as np
import chess
import torch.nn as nn
import torch
from pathlib import Path
import os


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

def board_position_list(board):
    position_list = [0] * 64
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color_offset = 6 if piece.color == chess.BLACK else 0
        position_list[square] = piece_type + color_offset
    return position_list

def board_position_list_one_hot(board):
    one_hot = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x = square // 8
        y = square % 8
        piece_type = piece.piece_type - 1
        color_offset = 6 if piece.color == chess.BLACK else 0
        one_hot[x, y, piece_type + color_offset] = 1
    return one_hot

def load_model(
        model_name: str = "fide_google_chess_model",
        with_weights: bool = True,
        **model_kwargs,
) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = Path(os.path.abspath('')).resolve() / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    return m

def piece_moved(position1, position2):
    affected_squares = []
    for i in range(64):
        if position1[i] != position2[i]:
            affected_squares.append(i)
    if len(affected_squares) > 2:
        for square in affected_squares:
            if position1[square] == 12 or position1[square] == 6:
                moved_from = square
            if position2[square] == 12 or position2[square] == 6:
                moved_to = square

            if position1[square] == 0:
                if position2[square] == 1:
                    moved_to = square
                    for square in affected_squares:
                        if position1[square] == 1:
                            moved_from = square
                elif position2[square] == 7:
                    moved_to = square
                    for square in affected_squares:
                        if position1[square] == 7:
                            moved_from = square
    else:
        if position2[affected_squares[0]] == 0:
            moved_from, moved_to = affected_squares[0], affected_squares[1]
        else:
            moved_from, moved_to = affected_squares[1], affected_squares[0]
    return moved_from, moved_to

def chess_bot(observation):
    game = Game(observation.board)
    moves = list(game.get_moves())

    for move in moves[:10]:
        g = Game(observation.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move

    for move in moves:
        if 'q' in move.lower():
            return move

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    game = Game(observation.board)
    moves = list(game.get_moves())
    board_position_one_hot_list = board_position_list_one_hot(chess.Board(observation.board))
    board_position_list_input = torch.from_numpy(board_position_one_hot_list).permute(2, 0, 1).unsqueeze(0).to(device)
    original_board_position_list = board_position_list(chess.Board(observation.board))
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
    piece_moved_pair = (move_from.item(), moved_to.item())

    res = float('inf')
    res_move = None
    for move in moves:
        # g = Game(observation.board)
        new_board = chess.Board(observation.board)
        new_board.push(chess.Move.from_uci(move))
        new_board_position_list = board_position_list(new_board)
        piece_from, piece_to = piece_moved(original_board_position_list, new_board_position_list)
        diff = abs(piece_from - piece_moved_pair[0]) + abs(piece_to - piece_moved_pair[1])

        if res > diff:
            res = diff
            res_move = move

        # g.apply_move(move)
        # if g.status == Game.CHECKMATE:
        #     return move

    # for move in moves:
    #     if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
    #         return move
    #
    # for move in moves:
    #     if 'q' in move.lower():
    #         return move

    # return random.choice(moves)
    return res_move

# class Observation:
#     def __init__(self):
#         self.board = Game().get_fen()
# chess_bot(Observation())