from Game import *
from DeepPlayer import *
from keras.optimizers import SGD
import model
import copy

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = None

    def train_epoch(self, games, epochs, generations):
        for _ in range(generations):
            board_data = []
            prob_data = []
            value_data = []
            
            for __ in range(games):
                # have both players use the same underlying model ("self-play")
                player_1 = DeepPlayer(self.model, PLAYER_1)
                player_2 = DeepPlayer(self.model, PLAYER_2)
                
                # prepare one additional game's worth of training data
                board_data_new, prob_data_new, value_data_new = self._generate_training_data(player_1, player_2)
                
                board_data += board_data_new
                prob_data += prob_data_new
                value_data += value_data_new

            bs = np.size(value_data)

            board_data = np.squeeze(np.array(board_data), axis = 1) # remove the excess dimension
            prob_data = np.squeeze(np.array(prob_data), axis = 1) # here as well
            value_data = np.array(value_data)
            
            # now can train the model
            self.model.model.fit(board_data, [prob_data, value_data], batch_size = bs, epochs = epochs)
            
    def _generate_training_data(self, player_1, player_2):
        # first need to generate the training data
        training_game = Game(player_1, player_2)
        res = training_game.play()

        # now can read out the history and convert it into training data
        prob_data = np.array(player_1.prob_history + player_2.prob_history)

        if res == PLAYER_1_WINS:
            print("player 1 won")
            value_data = np.concatenate((np.full(player_1.moves_played, 1.0), np.full(player_2.moves_played, -1.0)))
        elif res == PLAYER_2_WINS:
            print("player 2 won")
            value_data = np.concatenate((np.full(player_1.moves_played, -1.0), np.full(player_2.moves_played, 1.0)))
        elif res == DRAW:
            value_data = np.concatenate((np.full(player_1.moves_played, 0.0), np.full(player_2.moves_played, 0.0)))

        board_data = np.squeeze(np.stack(player_1.get_board_history() + player_2.get_board_history(), axis = 0), axis = 1)
        bs = np.size(value_data)
        
        return np.split(board_data, bs, axis = 0), np.split(prob_data, bs, axis = 0), np.split(value_data, bs, axis = 0)

    def setup(self):
        self.optimizer = SGD(lr = 1e-2, momentum = 0.9)
        cost_function = [model.prob_cost, model.value_cost]
        self.model.model.compile(optimizer = self.optimizer, loss = cost_function)

    def get_trained_model(self):
        return self.model
