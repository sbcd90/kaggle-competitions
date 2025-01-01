import chess.pgn
import os
from time import time
import h5py

import numpy as np
import pandas as pd

h5_folder = "../training_data"

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


def parse_file(directory, game_file):
    pgn = open(os.path.join(directory, game_file), "r")
    counter = 0
    startfile = time()
    filename = game_file.split(".")[0]
    train_input, moved_from, moved_to = [], [], []

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        counter += 1
        board = game.board()
        first = True

        for move in game.mainline_moves():
            if first:
                position1 = board_position_list(board)
                one_hot_position = board_position_list_one_hot(board)
                train_input.append(one_hot_position)
                first = False
            board.push(move)

            one_hot_position = board_position_list_one_hot(board)
            position2 = board_position_list(board)
            train_input.append(one_hot_position)
            piece_from, piece_to = piece_moved(position1, position2)
            moved_from.append(piece_from)
            moved_to.append(piece_to)

            position1 = position2

        if len(train_input) - len(moved_from) == 1:
            del train_input[-1]

    try:
        position = np.array(train_input)
        moved_from = np.array(moved_from)
        moved_from_one_hot = np.zeros((moved_from.size, 64))
        moved_from_one_hot[np.arange(moved_from.size), moved_from] = 1
        moved_to = np.array(moved_to)
        moved_to_one_hot = np.zeros((moved_to.size, 64))
        moved_to_one_hot[np.arange(moved_to.size), moved_to] = 1

        h5f = h5py.File(h5_folder + "/" + filename + ".h5", "w")
        h5f.create_dataset("input_position", data=position)
        h5f.create_dataset("moved_from", data=moved_from)
        h5f.create_dataset("moved_to", data=moved_to)
        h5f.close()
    except:
        print(50 * '-')
        print(50 * '-')
        print('ERROR IN {}, GAME {}'.format(game_file, counter))
        print(50 * '-')
        print(50 * '-')
    print(f"\n{filename} processed in {time() - startfile:0.3f} seconds")

def main(data_folder):
    for file in os.listdir(data_folder):
        df = pd.DataFrame(parse_file(data_folder, file))
        df.to_hdf(f"{h5_folder}/training_data.h5", key="df", mode="a")


if __name__ == "__main__":
    start_time = time()
    main(data_folder="../raw_data")
    print(f"Finished in {time() - start_time:0.3f} seconds")