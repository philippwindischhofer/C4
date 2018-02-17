import numpy as np
from constants import *
from termcolor import colored

class Board:

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.history = []

        # hold the position of the last stone that was successfully added
        self.last_move_col = 0
        self.last_move_row = 0
        self.last_player = PLAYER_2
        
        self.number_moves = 0
        self.status = NOT_TERMINATED
        self.board = np.full((rows, cols), FREE)
    
    def _print_board_header(self):
        for i in range(0, self.cols):
            print(colored(u'\u2502 ' + str(i) + ' ', 'grey'), end = '')
        print(colored(u'\u2502'), end = '\n')
        self._print_box_separator()

    def _print_box_header(self):
        print(colored(u'\u250C', 'grey'), end = '')
        for i in range(0, self.cols - 1):
            print(colored(u'\u2500\u2500\u2500\u252C', 'grey'), end = '')
        print(colored(u'\u2500\u2500\u2500\u2510', 'grey'), end = '\n')

    def _print_box_separator(self):
        print(colored(u'\u251C', 'grey'), end = '')
        for i in range(0, self.cols - 1):
            print(colored(u'\u2500\u2500\u2500\u253C', 'grey'), end = '')
        print(colored(u'\u2500\u2500\u2500\u2524', 'grey'), end = '\n')

    def _print_box_footer(self):
        print(colored(u'\u2514', 'grey'), end = '')
        for i in range(0, self.cols - 1):
            print(colored(u'\u2500\u2500\u2500\u2534', 'grey'), end = '')
        print(colored(u'\u2500\u2500\u2500\u2518', 'grey'), end = '\n')
        
    def print_board(self):
        self._print_box_header()
        self._print_board_header()
        for i in range(self.rows):
            for j in range(self.cols):
                print(colored(u'\u2502 ', 'grey'), end = '')
                if self.board[i, j] == PLAYER_1:
                    print(colored('' + u'\u25CF' + ' ', PLAYER_1_COLOR), end = '')
                elif self.board[i, j] == PLAYER_2:
                    print(colored('' + u'\u25CF' + ' ', PLAYER_2_COLOR), end = '')
                else:
                    print(colored('  ', 'grey'), end = '')
            print(colored(u'\u2502', 'grey'), end = '\n')
            if i < self.rows - 1:
                self._print_box_separator()
            else:
                self._print_box_footer()
        print('')
        
    def get_legal_moves(self):
        retval = []
        for col in range(self.cols):
            if self.board[0][col] == FREE:
                retval.append(col)
        return retval

    def get_opposite_player(self, player):
        if player == PLAYER_1:
            return PLAYER_2
        else:
            return PLAYER_1

    # returns the board position in the form required by the neural network
    def get_position(self):
        retval = np.full((1, 2, self.rows, self.cols), FREE) # first slice holds the position information for the player who has moved last, second slice for the player who will move next
        oplayer = self.get_opposite_player(self.last_player)
        retval[0][0] = (self.board == self.last_player) * 1.0        
        retval[0][1] = (self.board == oplayer) * 1.0
        return retval

    # returns the board position in a mirrored fashion, is just as valid (from the NN point of view) as the original setup
    def get_mirrored_position(self):
        retval = self.get_position()

        retval[0][0] = np.flip(retval[0][0], axis = 1)
        retval[0][1] = np.flip(retval[0][1], axis = 1)
        
        return retval

    def get_board_representation(self):
        return self.board.tostring()
        
    def place_stone(self, col):
        if col >= self.cols:
            return False
        
        row = 0
        while row < self.rows and self.board[row][col] == FREE:
            row += 1

        if row != 0:
            current_player = self.get_opposite_player(self.last_player) # the player whose turn it is now
            self.board[row - 1][col] = current_player

            # update all the statistics
            self.last_move_col = col
            self.last_move_row = row - 1
            self.number_moves += 1
            self.history.append(np.copy(self.board))

            # switch player for the next move
            self.last_player = current_player
            
            # check now if the game has ended
            self.status = self.check_win()
            
            return True
        else:
            return False

    # goes --->
    def _0_count(self, row, col, player):
        cnt = 0
        
        while col < self.cols:
            if self.board[row][col] == player:
                cnt += 1
                col += 1
            else:
                break
        return cnt
    # goes <---
    def _180_count(self, row, col, player):
        cnt = 0

        while col >= 0:
            if self.board[row, col] == player:
                cnt += 1
                col -= 1
            else:
                break
        return cnt

    def _90_count(self, row, col, player):
        cnt = 0
        
        while row < self.rows:
            if self.board[row][col] == player:
                cnt += 1
                row += 1
            else:
                break
        return cnt

    def _45_count(self, row, col, player):
        cnt = 0

        while row < self.rows and col < self.cols:
            if self.board[row][col] == player:
                cnt += 1
                row += 1
                col += 1
            else:
                break
        return cnt

    def _225_count(self, row, col, player):
        cnt = 0

        while row >= 0 and col >= 0:
            if self.board[row][col] == player:
                cnt += 1
                row -= 1
                col -= 1
            else:
                break
        return cnt
    
    def _315_count(self, row, col, player):
        cnt = 0
    
        while row >= 0 and col < self.cols:
            if self.board[row][col] == player:
                cnt += 1
                row -= 1
                col += 1
            else:
                break
        return cnt

    def _135_count(self, row, col, player):
        cnt = 0

        while row < self.rows and col >= 0:
            if self.board[row][col] == player:
                cnt += 1
                row += 1
                col -= 1
            else:
                break
        return cnt

    def _check_board_full(self):
        if self.number_moves == self.cols * self.rows:
            return True
        else:
            return False

    def check_win(self):
        r = self.last_move_row
        c = self.last_move_col
        p = self.last_player
        
        if np.max([self._0_count(r, c, p) + self._180_count(r, c, p) - 1,
                   self._45_count(r, c, p) + self._225_count(r, c, p) - 1,
                   self._135_count(r, c, p) + self._315_count(r, c, p) - 1, self._90_count(r, c, p)]) >= 4:
            return p
        elif self.number_moves == self.rows * self.cols:
            return DRAW
        
        return NOT_TERMINATED

    # computes the "reward" (i.e. win / lose) for the neural network, from the point of view of the player who is going to move, i.e. from the perspective of the current player
    # at the end of the day, this also sets the point of view of the neural network, that it is going to be trained on
    def get_reward(self):
        return -1.0 # due to the rules of connect-4, ONLY the player that was last active can ever win! -> very bad for the one who is going to move
