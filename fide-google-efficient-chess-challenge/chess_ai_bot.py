from Chessnut import Game
import chess


class ChessBot:
    def __init__(self):
        self.piece_values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}

    def is_safe_for_king(self, game, move):
        turn = True if game.get_fen().split()[1] == "w" else False
        board = chess.Board(game.get_fen())
        board.push(chess.Move.from_uci(move))

        king_square = board.king(turn)
        if board.is_attacked_by(not turn, king_square):
            return False
        return True

    def evaluate_move_heuristics(self, game, move, en_passant_moves, castling_moves):
        new_game = Game(game.get_fen())

        score = 0
        if new_game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            captured_piece = new_game.board.get_piece(Game.xy2i(move[2:4])).lower()
            if captured_piece:
                score += self.piece_values[captured_piece.lower()]

        if move in en_passant_moves:
            score += 3

        if move in castling_moves:
            if not self.is_safe_for_king(new_game, move):
                score += 15
            else:
                score += 10

        trial_game = Game(game.get_fen())
        trial_game.apply_move(move)
        if trial_game.status == new_game.CHECK:
            if trial_game.status != new_game.CHECKMATE:
                score += 5

        center_squares = ["d4", "e4", "d5", "e5"]
        if move[2:4] in center_squares:
            score += 1

        if new_game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            captured_piece = new_game.board.get_piece(Game.xy2i(move[2:4])).lower()
            if captured_piece in ["n", "b"]:
                score += 2

        trial_game = Game(game.get_fen())
        trial_game.apply_move(move)
        if trial_game.status == new_game.CHECKMATE:
            score += 100

        turn = True if game.get_fen().split()[1] == "w" else False
        board = chess.Board(game.get_fen())
        move_pos = chess.Move.from_uci(move)
        board.push(move_pos)
        if board.is_attacked_by(not turn, move_pos.to_square):
            score -= self.piece_values[chess.piece_symbol(board.piece_type_at(move_pos.to_square))] / 2

        return score

    def order_moves(self, game, moves, en_passant_moves, castling_moves):
        return sorted(moves, key=lambda move: self.evaluate_move_heuristics(game, move,
                                                                            en_passant_moves, castling_moves), reverse=True)

    def evaluate_board(self, game, castling_rights, en_passant_square):
        score = 0

        for i in range(64):
            piece = game.board.get_piece(i)
            if piece != ' ':
                value = self.piece_values[piece.lower()]
                if piece.isupper():
                    score += value
                else:
                    score -= value

        if "K" in castling_rights:
            score += 0.5
        if "Q" in castling_rights:
            score += 0.5
        if "k" in castling_rights:
            score -= 0.5
        if "q" in castling_rights:
            score -= 0.5

        if en_passant_square is not None:
            score += 0.1 if game.get_fen().split()[1] == "w"  else -0.1
        return score

    def get_castling_moves(self, game, castling_rights):
        moves = []
        if "K" in castling_rights and "O-O" in game.get_moves():
            moves.append("O-O")
        if "Q" in castling_rights and "O-O-O" in game.get_moves():
            moves.append("O-O-O")
        if "k" in castling_rights and "O-O" in game.get_moves():
            moves.append("O-O")
        if "q" in castling_rights and "O-O-O" in game.get_moves():
            moves.append("O-O-O")
        return moves

    def get_en_passant_moves(self, game, en_passant_square):
        moves = []
        if en_passant_square is not None:
            for move in game.get_moves():
                if move[2:4] == en_passant_square:
                    moves.append(move)
        return moves

    def mini_max_value(self, game, depth, castling_rights, en_passant_square, maximizing_player):
        alpha = float('-inf')
        beta = float('inf')
        if maximizing_player:
            return self.max_value(game, depth, castling_rights, en_passant_square, alpha, beta)
        else:
            return self.min_value(game, depth, castling_rights, en_passant_square, alpha, beta)


    def min_value(self, game, depth, castling_rights, en_passant_square, alpha, beta):
        if depth == 0 or game.status in {Game.CHECKMATE}:
            return self.evaluate_board(game, castling_rights, en_passant_square), None

        v = float('inf')
        best_move = None

        moves = list(game.get_moves())
        castling_moves = self.get_castling_moves(game, castling_rights)
        moves.extend(castling_moves)
        en_passant_moves = self.get_en_passant_moves(game, en_passant_square)
        moves.extend(en_passant_moves)
        moves = self.order_moves(game, moves, en_passant_moves, castling_moves)
        # try:
        #     moves = random.sample(moves, 10)
        # except ValueError as e:
        #     pass

        for move in moves[:2]:
            new_game = Game(game.get_fen())
            new_game.apply_move(move)
            new_score, _ = self.max_value(new_game, depth - 1, castling_rights, en_passant_square, alpha, beta)
            if new_score < v:
                v = new_score
                best_move = move
            if v <= alpha:
                return v, best_move
            beta = min(beta, v)

        return v, best_move

    def max_value(self, game, depth, castling_rights, en_passant_square, alpha, beta):
        if depth == 0 or game.status in {Game.CHECKMATE}:
            return self.evaluate_board(game, castling_rights, en_passant_square), None

        v = float('-inf')
        best_move = None

        moves = list(game.get_moves())
        castling_moves = self.get_castling_moves(game, castling_rights)
        moves.extend(castling_moves)
        en_passant_moves = self.get_en_passant_moves(game, en_passant_square)
        moves.extend(en_passant_moves)
        moves = self.order_moves(game, moves, en_passant_moves, castling_moves)
        # try:
        #     moves = random.sample(moves, 10)
        # except ValueError as e:
        #     pass

        for move in moves[:2]:
            new_game = Game(game.get_fen())
            new_game.apply_move(move)
            new_score, _ = self.min_value(new_game, depth - 1, castling_rights, en_passant_square, alpha, beta)
            if new_score > v:
                v = new_score
                best_move = move
            if v >= beta:
                return v, best_move
            alpha = max(alpha, v)

        return v, best_move

    def get_castling_rights(self, fen):
        fen_parts = fen.split()
        castling_rights = fen_parts[2]
        return set(castling_rights)

    def get_en_passant_square(self, fen):
        fen_parts = fen.split()
        en_passant_square = fen_parts[3]
        return en_passant_square if en_passant_square != "-" else None

    def get_move(self, board, depth=4):
        game = Game(board)
        castling_rights = self.get_castling_rights(game.get_fen())
        en_passant_square = self.get_en_passant_square(game.get_fen())
        _, best_move = self.mini_max_value(game, depth, castling_rights, en_passant_square, game.get_fen().split()[1] == "w")
        return best_move

def chess_bot(observation):
    # bot = ChessBot()
    # best_move = bot.get_move(observation.board, depth=2)
    # # print("hit here")
    # return best_move
    game = Game(observation.board)
    bot = ChessBot()
    en_passant_square = bot.get_en_passant_square(game.get_fen())
    en_passant_moves = bot.get_en_passant_moves(game, en_passant_square)
    castling_rights = bot.get_castling_rights(game.get_fen())
    castling_moves = bot.get_castling_moves(game, castling_rights)

    ordered_moves = bot.order_moves(game, list(game.get_moves()), en_passant_moves, castling_moves)
    return ordered_moves[0]
    # game = Game(observation.board)
    # print(game.move_history)
    # moves = list(game.get_moves())
    #
    # for move in moves[:10]:
    #     g = Game(observation.board)
    #     g.apply_move(move)
    #     if g.status == Game.CHECKMATE:
    #         return move
    #
    # for move in moves:
    #     if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
    #         return move
    #
    # for move in moves:
    #     if 'q' in move.lower():
    #         return move
    #
    # return random.choice(moves)