import numpy as np

class HumanPlayer:

    def move(self, board):
        valid = board.get_legal_moves()
        while True:
            move = int(input("specify your move: "))

            if move in valid:
                return move
            else:
                print("invalid choice")

    def reset(self):
        pass
