import pygame
import chess
import math
import torch.nn as nn
import torch
import numpy as np
from pathlib import Path
import os
from Chessnut import Game

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
        model_path = Path(os.path.abspath('')).resolve() / f"models/{model_name}.th"
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

def board_position_list(board):
    position_list = [0] * 64
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color_offset = 6 if piece.color == chess.BLACK else 0
        position_list[square] = piece_type + color_offset
    return position_list

class ChessGameUI:
    def __init__(self):
        self.X = 800
        self.Y = 800
        self.screen = pygame.display.set_mode((self.X, self.Y))
        pygame.init()

        self.WHITE = (255, 255, 255)
        self.GREY = (128, 128, 128)
        self.YELLOW = (204, 204, 0)
        self.BLUE = (50, 255, 255)
        self.BLACK = (0, 0, 0)

        self.board = chess.Board()

        self.pieces = {
            "p": pygame.image.load("./images/B_PAWN.png").convert(),
            "n": pygame.image.load("./images/B_KNIGHT.png").convert(),
            "b": pygame.image.load("./images/B_BISHOP.png").convert(),
            "r": pygame.image.load("./images/B_ROOK.png").convert(),
            "q": pygame.image.load("./images/B_QUEEN.png").convert(),
            "k": pygame.image.load("./images/B_KING.png").convert(),
            "P": pygame.image.load("./images/W_PAWN.png").convert(),
            "N": pygame.image.load("./images/W_KNIGHT.png").convert(),
            "B": pygame.image.load("./images/W_BISHOP.png").convert(),
            "R": pygame.image.load("./images/W_ROOK.png").convert(),
            "Q": pygame.image.load("./images/W_QUEEN.png").convert(),
            "K": pygame.image.load("./images/W_KING.png").convert(),
        }

    def update(self, screen, board):
        for i in range(64):
            piece = board.piece_at(i)
            if piece == None:
                pass
            else:
                screen.blit(self.pieces[str(piece)], ((i%8)*100,700-(i//8)*100))
        for i in range(7):
            i += 1
            pygame.draw.line(screen, self.WHITE, (0, i*100), (800, i*100))
            pygame.draw.line(screen, self.WHITE, (i*100, 0), (i*100, 800))
        pygame.display.flip()

    def __piece_moved(self, position1, position2):
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

    def agent(self, board: chess.Board) -> chess.Move:
        # game = Game(board.fen())
        # moves = list(game.get_moves())
        #
        # for move in moves[:10]:
        #     g = Game(board.fen())
        #     g.apply_move(move)
        #     if g.status == Game.CHECKMATE:
        #         return chess.Move.from_uci(move)
        #
        # for move in moves:
        #     if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
        #         return chess.Move.from_uci(move)
        #
        # for move in moves:
        #     if 'q' in move.lower():
        #         return chess.Move.from_uci(move)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        board_position_one_hot_list = board_position_list_one_hot(board).tolist()
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
        predicted_pieces = {"moved_from": move_from.item(), "moved_to": moved_to.item()}

        res = float('inf')
        res_move = None
        all_moves = board.legal_moves
        for move in all_moves:
            temp_board = chess.Board(board.fen())
            temp_board_position_list_init = board_position_list(temp_board)
            temp_board.push(move)
            temp_board_position_list_after_move = board_position_list(temp_board)
            piece_from, piece_to = self.__piece_moved(temp_board_position_list_init, temp_board_position_list_after_move)
            diff = abs(piece_from - predicted_pieces["moved_from"]) + abs(piece_to - predicted_pieces["moved_to"])

            if res > diff:
                res = diff
                res_move = move
        return res_move

    def main_one_agent(self, agent_color: chess.Color):
        self.screen.fill(self.BLACK)
        pygame.display.set_caption("Chess")
        
        index_moves = []
        status = True
        while status:
            self.update(self.screen, self.board)
            if self.board.turn == agent_color:
                self.board.push(self.agent(self.board))
                self.screen.fill(self.BLACK)
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        status = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.screen.fill(self.BLACK)
                        pos = pygame.mouse.get_pos()

                        square = (math.floor(pos[0]/100), math.floor(pos[1]/100))
                        index = (7 - square[1]) * 8 + square[0]

                        if index in index_moves:
                            move = moves[index_moves.index(index)]
                            self.board.push(move)

                            index = None
                            index_moves = []
                        else:
                            piece = self.board.piece_at(index)
                            if piece == None:
                                pass
                            else:
                                all_moves = list(self.board.legal_moves)
                                moves = []
                                for m in all_moves:
                                    if m.from_square == index:
                                        moves.append(m)

                                        t = m.to_square
                                        tx1 = 100 * (t % 8)
                                        ty1 = 100 * (7 - t // 8)

                                        pygame.draw.rect(self.screen, self.BLUE, pygame.Rect(tx1, ty1, 100, 100), 5)
                                index_moves = [a.to_square for a in moves]
            if self.board.outcome() != None:
                print(self.board.outcome())
                status = False
                print(self.board)

        pygame.quit()

def main():
    chess_game_ui = ChessGameUI()
    chess_game_ui.main_one_agent(chess.BLACK)

if __name__ == "__main__":
    main()
