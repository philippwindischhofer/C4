from model import C4Model
from constants import *
import numpy as np
import copy

class DeepPlayer:
    def __init__(self, model, player):
        self.model = model
        self.c_exp = 0.5 # sets the exploration fraction during the tree search
        self.MCTS_iter = 100
        self.player = player # need to know on which side we are!

        # keep track of the probabilities and the game situations that were encountered
        self.prob_history = []
        self.board_history = []
        self.moves_played = 0

        # this holds the root node of the tree
        self.tree = None

    def move(self, board):
        # first, backup the board situation on which this move is going to be based
        self.board_history.append(copy.deepcopy(board))
        
        # invoke the model here and compute the move, given the board as input data (also add the tree improvement)
        raw_eval = self.model.model.predict(x = board.get_position(), verbose = 0, batch_size = 1)
        
        prior = raw_eval[0][0]
        value = raw_eval[1][0][0]
        
        # start up the tree improvement: build the root node of the tree
        self.tree = GameState(board, prior, value)

        # now run the MCTS for a fixed number of times
        for i in range(self.MCTS_iter):
            self._search(self.tree, self.model)

        # read out the final tree-improved answer for the moves to take (and their probabilities)
        action, prob = self.tree.get_moves()

        # now, also backup the tree-improved probabilities for the next move (insert a zero if a move is illegal)
        prob_backup = np.zeros(board.cols)
        for i in range(len(prob)):
            prob_backup[action[i]] = prob[i]

        self.prob_history.append(prob_backup)
                
        # now sample a move from the improved distribution
        self.moves_played += 1
        action_played = np.random.choice(action, p = prob)

        # retain the entire tree, but make the actually played node the new root node

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

    # does *one* simulation run (in the sense of MCTS) starting from the node "state" and updates the tree
    # NOTE: all rewards are from the point of view of the player who is going to move next -> s.t. at every step in the recursion, can look at the MAXIMUM of the U-values (as exploration-corrected, averaged v-values)
    def _search(self, state, model):

        #print("running search")
        
        if state.board.status != NOT_TERMINATED:
            # this game is actually already done, can return the correct result, as per the rules
            return -state.board.get_reward()

        # game is not yet over, need to search through the tree to find the one with the highest U-value
        max_U = -float("inf")
        best_action = None

        # compute for the normalization N_tot = sqrt(N_a)
        N_tot = 0
        for action in state.actions:
            N_tot += action.N
        
        for action in state.actions:
            # compute the U values for the local actions
            cur_U = action.Q + self.c_exp * action.P * np.sqrt(N_tot) / (1 + action.N)
            if cur_U > max_U:
                max_U = cur_U
                best_action = action

        # "best_action" now contains the action with the highest U-value
        if best_action.linked_state() is None:
            # have reached a leaf node of the tree, extend the tree in this direction
            # first, need to make a copy of the current board, then apply this action
            new_board = copy.deepcopy(state.board)
            new_board.place_stone(best_action.action)

            new_eval = self.model.model.predict(x = new_board.get_position(), verbose = 0, batch_size = 1)
            prior = new_eval[0][0]
            v = new_eval[1][0][0]

            # then create the new game state ...
            new_state = GameState(new_board, prior, v)

            # ... and integrate it into the tree
            best_action.link(new_state)

            return -v
        else:
            # the tree continues here, can just go on
            v = self._search(best_action.linked_state(), model) # this is the value resulting from the tree search

            # update the link of the best action that was just taken
            best_action.Q = (best_action.Q * best_action.N + v) / (best_action.N + 1)
            best_action.N += 1

            return -v

# represents a valid, legal action that can be taken, connecting two board positions
class GameAction:
    def __init__(self, action):
        # for a new link, initialize them with the standard values
        self.Q = 0
        self.N = 0
        self.P = 0
        self.action = action
        self.new_state = None

    def link(self, new_state):
        self.new_state = new_state

    def linked_state(self):
        return self.new_state
        
# represents a certain (momentary) state in the game, i.e. a board position, together with the links to all other positions that can be reached from here
class GameState:
    def __init__(self, board, prior, value):
        # for a new tree node, set the board, prior probabilities and values coming from the neural network
        self.board = board
        self.value = value

        # for a new tree node, also generate all possible (empty) actions that can be taken ...
        self.actions = []
        for action in board.get_legal_moves():
            self.actions.append(GameAction(action))
            
        # ... and initialize them with the prior probabilities
        for action in self.actions:
            action.P = prior[action.action]

        # NOTE: Q and N stay initialized with zero!

    # returns the array of the possible moves, and their probabilities (given by the frequency with which they were visited)
    def get_moves(self):
        N_tot = 0
        prob_out = []
        action_out = []

        for action in self.actions:
            N_tot += action.N

        for action in self.actions:
            prob_out.append(action.N / N_tot)
            action_out.append(action.action)

        return action_out, prob_out
