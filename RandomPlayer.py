import random

class RandomPlayer:

    def move(self, board):
        moves = board.get_legal_moves()
        return random.choice(moves)

    def reset(self):
        pass
