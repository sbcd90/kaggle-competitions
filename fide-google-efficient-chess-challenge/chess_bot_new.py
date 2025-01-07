from Chessnut import Game
import chess
import requests

def board_position_list(board):
    position_list = [0] * 64
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color_offset = 6 if piece.color == chess.BLACK else 0
        position_list[square] = piece_type + color_offset
    return position_list

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

    response = requests.post("https://sbcd90.pythonanywhere.com", headers={"Content-Type": "text/plain"},
                             data=observation.board)
    predicted_pieces = response.json()

    original_board_position_list = board_position_list(chess.Board(observation.board))
    res = float('inf')
    res_move = None
    for move in moves:
        new_board = chess.Board(observation.board)
        new_board.push(chess.Move.from_uci(move))
        new_board_position_list = board_position_list(new_board)
        piece_from, piece_to = piece_moved(original_board_position_list, new_board_position_list)
        diff = abs(piece_from - predicted_pieces["moved_from"]) + abs(piece_to - predicted_pieces["moved_to"])

        if res > diff:
            res = diff
            res_move = move
    return res_move

# class Observation:
#     def __init__(self):
#         self.board = Game().get_fen()
# chess_bot(Observation())