from board import *
from constants import *
from RandomPlayer import *
from Game import *
from model import *
from DeepPlayer import *
from Trainer import *
from utils import Evaluator
from optparse import OptionParser
import sys

parser = OptionParser()
parser.add_option("-t", "--train", action = "store_true", dest = "train", help = "train a new model or resume training of an already existing model [if specified with -f]")
parser.add_option("-f", "--file", action = "store", type = "string", dest = "model_file", help = "specifies a model to use for training, benchmarking or manual play")
parser.add_option("-c", "--computer", action = "store_true", dest = "computer_opponent", help = "launches a new game against a model [specified with -f]")
parser.add_option("-m", "--manual", action = "store_true", dest = "human_opponent", help = "launches a new game against a human opponent")
parser.add_option("-b", "--benchmark", action = "store_true", dest = "benchmark", help = "benchmarks a model [specified with -f] against a random player")

(options, args) = parser.parse_args()

if options.train:
    from config import *
else:
    from config_manual_play import *

model_config = ModelConfig()
player_config = DeepPlayerConfig()

model = C4Model(model_config)
model.build()

if options.train:
    # make a trainer to train the model on self-play data
    if options.model_file != None:
        print("resuming training of " + options.model_file)
        model.load(folder = 'models', filename = options.model_file) # keep going from the previous state of the art
    else:
        print("start training from scratch")
    trainer = Trainer(model, player_config)
    trainer.setup()
    trainer.train_epoch(games = 100, training_epochs = 50, generations = 50)
elif options.computer_opponent:
    # evaluate the trained model against a human player
    if options.model_file != None:
        model.load(folder = 'models', filename = options.model_file)
    else:
        print("Error: require -f to specify the model")
        sys.exit(1)
    Evaluator.combat_human(model, 1, player_config)
elif options.human_opponent:
    Evaluator.combat_humans(1)
elif options.benchmark:
    # evaluate the trained model against a random player
    if options.model_file != None:
        model.load(folder = 'models', filename = options.model_file)
    else:
        print("Error: require -f to specify the model")
        sys.exit(1)
    wins, draws = Evaluator.combat_random(model, 100, player_config)
    print("wins = " + str(wins))
