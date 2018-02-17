from DeepPlayer import *
from RandomPlayer import *
from HumanPlayer import *
from constants import *
from Game import *

class Evaluator:
    # returns the fraction of wins of model_1 against model_2
    @staticmethod
    def combat_model(model_1, model_2, iterations, player_config):
        tree_1 = MCTS(model_1)
        tree_2 = MCTS(model_2)
        player_1 = DeepPlayer(model_1, tree_1, player_config)
        player_2 = DeepPlayer(model_2, tree_2, player_config)
        
        return Evaluator.combat(player_1, player_2, iterations, False)
    
    # returns the fraction of wins of model against a random player
    @staticmethod
    def combat_random(model, iterations, player_config):
        tree = MCTS(model)
        player_1 = DeepPlayer(model, tree, player_config)
        player_2 = RandomPlayer()
        
        return Evaluator.combat(player_1, player_2, iterations, False)

    @staticmethod
    def combat_human(model, iterations, player_config):
        tree = MCTS(model)
        player_2 = DeepPlayer(model, tree, player_config)
        player_1 = HumanPlayer()

        return Evaluator.combat(player_1, player_2, iterations, True)

    @staticmethod
    def combat_humans(iterations):
        player_1 = HumanPlayer()
        player_2 = HumanPlayer()

        return Evaluator.combat(player_1, player_2, iterations, True)
    
    # has players 1 and 2 play against each other
    @staticmethod
    def combat(player_1, player_2, iterations, verbose):
        player_1_wins = 0
        player_2_wins = 0
        draws = 0
        
        for i in range(iterations):
            
            player_1.reset()
            player_2.reset()
            game = Game(player_1, player_2)
            res = game.play(verbose)
        
            if res == PLAYER_1_WINS:
                player_1_wins += 1
            elif res == PLAYER_2_WINS:
                player_2_wins += 1
            elif res == DRAW:
                draws += 1
            
        return player_1_wins / iterations, draws / iterations
