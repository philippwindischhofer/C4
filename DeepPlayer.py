from model import C4Model
from constants import *
import numpy as np
import copy

class DeepPlayer:
    def __init__(self, model, tree):
        self.model = model
        self.MCTS_iter = 40

        # keep track of the probabilities and the game situations that were encountered
        self.prob_history = []
        self.board_history = []
        self.moves_played = 0

        # this holds the root node of the tree
        self.tree = tree

    def move(self, board):
        # first, backup the board situation on which this move is going to be based
        self.board_history.append(copy.deepcopy(board))
    
        # find the best move by MCTS:
        # first, run the MCTS for a fixed number of times
        for i in range(self.MCTS_iter):
            self.tree.search(board)

        # second, read out the final tree-improved answer for the moves to take (and their probabilities)
        action, prob = self.tree.get_moves(board)

        # now, also backup the tree-improved probabilities for the next move (insert a zero if a move is illegal)
        prob_backup = np.zeros(board.cols)
        for i in range(len(prob)):
            prob_backup[action[i]] = prob[i]

        self.prob_history.append(prob_backup)
                
        # now sample a move from the improved distribution
        self.moves_played += 1

        if self.moves_played >= 0:
            action_played = action[np.argmax(prob)]
        else:
            action_played = np.random.choice(action, p = prob)

        return action_played

    def get_board_history(self):
        retval = []
        for board in self.board_history:
            retval.append(board.get_position())
        return retval

    def reset(self):
        self.prob_history = []
        self.board_history = []
        self.moves_played = 0
        self.tree.reset()

class MCTS:

    def __init__(self, model):
        self.model = model

        self.c_exp = 1.0 # sets the exploration fraction during the tree search

        # properties associated to the positions
        self.boards = {} # holds all the boards ...
        self.values = {} # ... and their values

        # properties associated to the moves (actions) that link positions
        self.Q = {}
        self.N = {}
        self.P = {} # ... the prior probabilities computed by the network

    def reset(self):
        # properties associated to the positions
        self.boards = {} # holds all the boards ...
        self.values = {} # ... and their values

        # properties associated to the moves (actions) that link positions
        self.Q = {}
        self.N = {}
        self.P = {} # ... the prior probabilities computed by the network
        
    # does *one* simulation run (in the sense of MCTS) starting from the node "state" and updates the tree
    # NOTE: all rewards are from the point of view of the player who is going to move next -> s.t. at every step in the recursion, can look at the MAXIMUM of the U-values (as exploration-corrected, averaged v-values)
    def search(self, board):
        
        if board.status != NOT_TERMINATED:
            # this game is actually already done, can return the correct result, as per the rules
            return -board.get_reward()

        # prepare the key to look up this position in the tree
        key = board.get_board_representation()
        valid = board.get_legal_moves()
        
        # sit on a leaf node
        if key not in self.boards:
            # evaluate this position using the model associated with this tree ...
            new_eval = self.model.model.predict(x = board.get_position(), verbose = 0, batch_size = 1)
            prior = new_eval[0][0]
            v = new_eval[1][0][0]

            # ... add it to the tree ...
            self.boards[key] = copy.deepcopy(board)
            self.values[key] = v

            # ... and initialize all the links going to other positions
            for action in valid:
                self.P[(key, action)] = prior[action]
                self.Q[(key, action)] = 0
                self.N[(key, action)] = 0
            
            return -v            

        # game is not yet over, need to search through the tree to find the one with the highest U-value
        max_U = -float("inf")
        best_action = None

        # compute for the normalization N_tot = sqrt(N_a)
        N_tot = 0
        for action in valid:
            N_tot += self.N[(key, action)]
        
        for action in valid:
            # compute the U values for the locally available actions
            cur_U = self.Q[(key, action)] + self.c_exp * self.P[(key, action)] * np.sqrt(N_tot) / (1 + self.N[(key, action)])
            if cur_U > max_U:
                max_U = cur_U
                best_action = action

        # follow the best path
        new_board = copy.deepcopy(board)
        new_board.place_stone(best_action)
                
        # the tree continues here, can just go on
        v = self.search(new_board) # this is the value resulting from the tree search

        # update the link of the best action that was just taken
        self.Q[(key, best_action)] = (self.Q[(key, best_action)] * self.N[(key, best_action)] + v) / (self.N[(key, best_action)] + 1)
        self.N[(key, best_action)] += 1
        
        return -v

        # returns the array of the possible moves, and their probabilities (given by the frequency with which they were visited)
    def get_moves(self, board):
        N_tot = 0
        prob_out = []
        action_out = []

        key = board.get_board_representation()
        valid = board.get_legal_moves()
        
        for action in valid:
            N_tot += self.N[(key, action)]

        for action in valid:
            prob_out.append(self.N[(key, action)] / N_tot)
            action_out.append(action)

        return action_out, prob_out

    
