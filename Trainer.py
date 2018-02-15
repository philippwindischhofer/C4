from Game import *
from DeepPlayer import *
from keras.optimizers import SGD
import model
import copy
from utils import Evaluator

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.tree = MCTS(model)

    def train_epoch(self, games, training_epochs, generations):
        board_data_acc = []
        prob_data_acc = []
        value_data_acc = []
        
        for _ in range(generations):
             
            for __ in range(games):
                # have both players use the same underlying model ("self-play")
                # the only reason that need two objects here is to keep track of their respective moves
                player_1 = DeepPlayer(self.model, self.tree)
                player_2 = DeepPlayer(self.model, self.tree)
                 
                # prepare one additional game's worth of training data
                print("generating training data")
                board_data_new, prob_data_new, value_data_new = self._generate_training_data(player_1, player_2)
                 
                board_data_acc += board_data_new
                prob_data_acc += prob_data_new
                value_data_acc += value_data_new

                print("generated " + str(len(value_data_new)) + " moves, total accumulated dataset = " + str(len(value_data_acc)) + " moves")

            bs = np.size(value_data_acc)

            board_data = np.squeeze(np.array(board_data_acc), axis = 1) # remove the excess dimension
            prob_data = np.squeeze(np.array(prob_data_acc), axis = 1) # here as well
            value_data = np.array(value_data_acc)

            self.model.save() # save the old model before the new training step
            old_model = C4Model(self.model.config) # create a second one with the same configuration
            old_model.build()
            old_model.load()
            
            # now can train the model
            self.model.model.fit(board_data, [prob_data, value_data], batch_size = bs, epochs = training_epochs)

            # now need to compare the old and the new model against each other to decide which one to keep
            print("Evaluating trained model")
            wins, draws = Evaluator.combat_model(self.model, old_model, 40)
            if wins < 0.55:
                # the trained model works not significantly better than the older one, thus keep the older one and retry with more training data
                self.model = old_model
                print("wins = " + str(wins) + " -> keeping old model")
            else:
                print("wins = " + str(wins) + " -> keeping trained model")
                self.model.save(filename = 'best.tar') # save it as the new current best model
                self.tree.reset() # reset the tree, since now the neural network has changed

                # reset also the training data
                board_data_acc = []
                prob_data_acc = []
                value_data_acc = []
                
    def _generate_training_data(self, player_1, player_2):
        # first need to generate the training data
        training_game = Game(player_1, player_2)
        res = training_game.play()

        # now can read out the history and convert it into training data
        prob_data = np.array(player_1.prob_history + player_2.prob_history)

        if res == PLAYER_1_WINS:
            value_data = np.concatenate((np.full(player_1.moves_played, 1.0), np.full(player_2.moves_played, -1.0)))
        elif res == PLAYER_2_WINS:
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
