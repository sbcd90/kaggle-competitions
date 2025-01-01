import chess.pgn
import numpy as np

def position_list_one_hot(self):
    '''method added to the python-chess library for faster
    conversion of board to one hot encoding. Resulted in 100%
    increase in speed by bypassing conversion to fen() first.
    '''
    builder = []
    builder_append = builder.append
    for square in chess.SQUARES_180:
        mask = chess.BB_SQUARES[square]

        if not self.occupied & mask:
            builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif bool(self.occupied_co[chess.WHITE] & mask):
            if self.pawns & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif self.knights & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif self.bishops & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif self.rooks & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif self.queens & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif self.kings & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        elif self.pawns & mask:
            builder.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.knights & mask:
            builder.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.bishops & mask:
            builder.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.rooks & mask:
            builder.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.queens & mask:
            builder.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif self.kings & mask:
            builder.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    return builder

def position_list(self):
    '''same as position_list_one_hot except this is converts pieces to
    numbers between 1 and 12. Used for piece_moved function'''
    builder = []
    builder_append = builder.append
    for square in chess.SQUARES_180:
        mask = chess.BB_SQUARES[square]

        if not self.occupied & mask:
            builder_append(0)
        elif bool(self.occupied_co[chess.WHITE] & mask):
            if self.pawns & mask:
                builder_append(7)
            elif self.knights & mask:
                builder_append(8)
            elif self.bishops & mask:
                builder_append(9)
            elif self.rooks & mask:
                builder_append(10)
            elif self.queens & mask:
                builder_append(11)
            elif self.kings & mask:
                builder_append(12)
        elif self.pawns & mask:
            builder_append(1)
        elif self.knights & mask:
            builder_append(2)
        elif self.bishops & mask:
            builder_append(3)
        elif self.rooks & mask:
            builder_append(4)
        elif self.queens & mask:
            builder_append(5)
        elif self.kings & mask:
            builder_append(6)

    return builder

chess.BaseBoard.position_list_one_hot = position_list_one_hot
chess.BaseBoard.position_list = position_list


def board_position_list(board):
    position_list = [0] * 64
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color_offset = 6 if piece.color == chess.BLACK else 0
        position_list[square] = piece_type + color_offset
    return position_list

def board_position_list_one_hot(board):
    one_hot = np.zeros((64, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type - 1
        color_offset = 6 if piece.color == chess.BLACK else 0
        one_hot[square, piece_type + color_offset] = 1
    return one_hot.flatten()

def main():
    board = chess.Board()
    custom_one_hot_encoding = board_position_list_one_hot(board)
    standard_one_hot_encoding = board.position_list_one_hot()

    print("Are they identical?", custom_one_hot_encoding == standard_one_hot_encoding)

    custom_encoding = board_position_list(board)
    standard_encoding = board.position_list()
    print("Are they identical?", custom_encoding == standard_encoding)

if __name__ == "__main__":
    main()

