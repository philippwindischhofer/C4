from board import *

class Game:

    def __init__(self, player_1, player_2):
        self.board = Board(GAME_ROWS, GAME_COLS)
        # make sure the players start from scratch
        player_1.reset()
        player_2.reset()
        self.player_1 = player_1
        self.player_2 = player_2

    # go through the game and return the winner
    def play(self):
        while True:
            # player 1 is always the first one to move
            self.board.place_stone(self.player_1.move(self.board))
            #self.board.print_board()
            winner = self.board.status
            if winner != NOT_TERMINATED:
                break

            # then comes player 2
            self.board.place_stone(self.player_2.move(self.board))
            #self.board.print_board()
            winner = self.board.status
            if winner != NOT_TERMINATED:
                break

        return winner
